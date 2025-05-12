import gym
import numpy as np
import random
import pygame
import cv2
import os

# --- Initialize Pygame Assets ---
def load_assets():
    # Create a gradient sky background
    background = pygame.Surface((600, 200))
    for y in range(200):
        # Gradient from light blue to darker blue
        color = (135, 206, 235 - y // 2)
        pygame.draw.line(background, color, (0, y), (600, y))

    # Draw green ground with a pattern
    pygame.draw.rect(background, (34, 139, 34), (0, 150, 600, 50))
    for x in range(0, 600, 20):
        pygame.draw.line(background, (0, 100, 0), (x, 150), (x + 10, 160), 2)

    # Load player and opponent character images
    player_img = pygame.image.load("assets/dogconcept.png").convert_alpha()
    opponent_img = pygame.image.load("assets/templar_knight.png").convert_alpha()

    # Resize images to fit the game dimensions
    player_img = pygame.transform.scale(player_img, (40, 60))
    opponent_img = pygame.transform.scale(opponent_img, (80, 60))

    # Load cloud images for the background
    cloud_img = pygame.image.load("assets/cloud.png").convert_alpha()
    cloud_img = pygame.transform.scale(cloud_img, (100, 50))

    return background, player_img, opponent_img, cloud_img

# --- Load Sound Effects ---
def load_sounds():
    """
    Load sound effects for punches, kicks, and special moves.
    """
    pygame.mixer.init()  # Initialize the mixer module
    punch_sound = pygame.mixer.Sound("assets/punch.wav")
    kick_sound = pygame.mixer.Sound("assets/kick.wav")
    special_sound = pygame.mixer.Sound("assets/special.wav")
    return punch_sound, kick_sound, special_sound

# --- Custom Fighting Game Environment ---
class FightingGameEnv(gym.Env):
    def __init__(self, render_mode=False, record_video=False, video_path="gameplay.mp4"):
        super(FightingGameEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(9)  # [0: Stay, 1: Left, 2: Right, 3: Punch, 4: Kick, 5: Block, 6: Dodge, 7: Special Move, 8: Jump]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)  # Normalized state with jump

        # Constants
        self.MAX_HEALTH = 100
        self.MAX_STAMINA = 100
        self.MOVEMENT_SPEED = 10
        self.PUNCH_DAMAGE = 10
        self.KICK_DAMAGE = 15
        self.SPECIAL_MOVE_DAMAGE = 25
        self.STAMINA_COST = {
            3: 10,  # Punch
            4: 15,  # Kick
            5: 5,   # Block
            6: 10,  # Dodge
            7: 30,  # Special Move
            8: 10   # Jump
        }
        self.WINDOW_WIDTH = 600
        self.WINDOW_HEIGHT = 200
        self.opponent_aggression = 0.6  # Default aggression for curriculum learning

        self.render_mode = render_mode
        self.record_video = record_video
        self.video_path = video_path
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("Fighting Game Visualization")
            self.clock = pygame.time.Clock()
            self.background, self.player_img, self.opponent_img, self.cloud_img = load_assets()
            self.punch_sound, self.kick_sound, self.special_sound = load_sounds()

        if self.record_video:
            # Ensure video path is absolute and directory exists
            self.video_path = os.path.abspath(video_path)
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            print(f"Initializing video recording at: {self.video_path}")
            self.video_writer = cv2.VideoWriter(
                self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            )
            if not self.video_writer.isOpened():
                print(f"Error: Failed to initialize video writer for {self.video_path}")

        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.player_health = self.MAX_HEALTH
        self.opponent_health = self.MAX_HEALTH
        self.player_stamina = self.MAX_STAMINA
        self.opponent_stamina = self.MAX_STAMINA
        self.player_pos = 150
        self.opponent_pos = 450
        self.player_on_ground = True
        self.opponent_on_ground = True
        return self._get_state()

    def _get_state(self):
        """
        Get the current state of the environment.
        """
        return np.array([
            self.player_health / self.MAX_HEALTH,
            self.opponent_health / self.MAX_HEALTH,
            self.player_stamina / self.MAX_STAMINA,
            self.opponent_stamina / self.MAX_STAMINA,
            self.player_pos / self.WINDOW_WIDTH,
            self.opponent_pos / self.WINDOW_WIDTH,
            abs(self.player_pos - self.opponent_pos) / self.WINDOW_WIDTH,
            1 if self.player_pos < self.opponent_pos else -1,
            1 if self.player_on_ground else 0
        ], dtype=np.float32)

    def step(self, action):
        """
        Execute one step in the environment.
        """
        reward = -2  # Penalize standing still
        prev_state = self._get_state()  # Log previous state

        # Player Actions
        if action == 1:  # Move Left
            self.player_pos = max(0, self.player_pos - self.MOVEMENT_SPEED)
        elif action == 2:  # Move Right
            self.player_pos = min(self.WINDOW_WIDTH - 40, self.player_pos + self.MOVEMENT_SPEED)
        elif action == 3:  # Punch
            if self.player_stamina >= self.STAMINA_COST[3] and abs(self.player_pos - self.opponent_pos) <= 50:
                self.opponent_health -= self.PUNCH_DAMAGE
                self.player_stamina -= self.STAMINA_COST[3]
                reward += 15
                self.punch_sound.play()
        elif action == 4:  # Kick
            if self.player_stamina >= self.STAMINA_COST[4] and abs(self.player_pos - self.opponent_pos) <= 70:
                self.opponent_health -= self.KICK_DAMAGE
                self.player_stamina -= self.STAMINA_COST[4]
                reward += 20
                self.kick_sound.play()
        elif action == 5:  # Block
            if self.player_stamina >= self.STAMINA_COST[5]:
                self.player_stamina -= self.STAMINA_COST[5]
                reward += 5
        elif action == 6:  # Dodge
            if self.player_stamina >= self.STAMINA_COST[6]:
                self.player_stamina -= self.STAMINA_COST[6]
                reward += 10
        elif action == 7:  # Special Move
            if self.player_stamina >= self.STAMINA_COST[7] and abs(self.player_pos - self.opponent_pos) <= 50:
                self.opponent_health -= self.SPECIAL_MOVE_DAMAGE
                self.player_stamina -= self.STAMINA_COST[7]
                reward += 30
                self.special_sound.play()
        elif action == 8:  # Jump
            if self.player_stamina >= self.STAMINA_COST[8] and self.player_on_ground:
                self.player_on_ground = False
                self.player_stamina -= self.STAMINA_COST[8]
                reward += 5  # Encourage jumping for positioning

        # Opponent AI (stochastic behavior)
        if abs(self.opponent_pos - self.player_pos) > 50:
            if random.random() > (1 - self.opponent_aggression):  # Curriculum learning
                if self.opponent_pos > self.player_pos:
                    self.opponent_pos -= self.MOVEMENT_SPEED
                else:
                    self.opponent_pos += self.MOVEMENT_SPEED
        elif abs(self.opponent_pos - self.player_pos) <= 50 and random.random() > (1 - self.opponent_aggression):
            if self.opponent_stamina >= 10 and self.player_on_ground:  # Jump reduces damage
                self.player_health -= 10
                self.opponent_stamina -= 10

        # Simulate landing after jump
        if not self.player_on_ground:
            self.player_on_ground = True

        # Stamina Regeneration
        self.player_stamina = min(self.MAX_STAMINA, self.player_stamina + 5)
        self.opponent_stamina = min(self.MAX_STAMINA, self.opponent_stamina + 5)

        # Win/Loss Rewards
        if self.opponent_health <= 0:
            reward += 50  # Higher winning reward
        if self.player_health <= 0:
            reward -= 50  # Higher losing penalty

        # Check if game over
        done = self.player_health <= 0 or self.opponent_health <= 0
        next_state = self._get_state()

        # Log transition for debugging (1% of transitions)
        if random.random() < 0.01:
            print(f"Transition: Action={action}, Reward={reward}, Prev_State={prev_state}, Next_State={next_state}")

        if self.render_mode:
            self.render()

        return next_state, reward, done, {}

    def render(self):
        """
        Render the environment using Pygame.
        """
        self.screen.blit(self.background, (0, 0))  # Draw background
        self.screen.blit(self.player_img, (self.player_pos, 90 if self.player_on_ground else 70))  # Adjust for jump
        self.screen.blit(self.opponent_img, (self.opponent_pos, 90))
        pygame.draw.rect(self.screen, (0, 255, 0), (50, 20, self.player_health * 2, 15))  # Player health bar
        pygame.draw.rect(self.screen, (255, 0, 0), (350, 20, self.opponent_health * 2, 15))  # Opponent health bar
        pygame.draw.rect(self.screen, (0, 0, 255), (50, 40, self.player_stamina * 2, 10))  # Player stamina bar
        pygame.draw.rect(self.screen, (0, 0, 255), (350, 40, self.opponent_stamina * 2, 10))  # Opponent stamina bar
        pygame.display.flip()
        self.clock.tick(30)

        if self.record_video:
            # Capture frame and write to video
            frame = pygame.surfarray.array3d(self.screen)
            frame = frame.transpose([1, 0, 2])  # Convert to OpenCV format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)

    def close(self):
        """
        Clean up resources.
        """
        if self.record_video and hasattr(self, 'video_writer'):
            self.video_writer.release()
            print(f"Video saved at: {self.video_path}")
        pygame.quit()