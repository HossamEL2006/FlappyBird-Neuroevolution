"""
##############################################################################
#                                                                            #
#                       © 2024 HossamEL2006                                  #
#                                                                            #
#   You are permitted to use, copy, and modify this file for personal,       #
#   non-commercial purposes only. Redistribution of any modified or          #
#   unmodified version of this file, especially for commercial purposes,     #
#   is not allowed without explicit permission from the author.              #
#                                                                            #
##############################################################################

This module implements a Flappy Bird game using Pygame, along with a simple 
neuroevolution algorithm that trains birds to play the game using neural networks.

This script was mostly documented and commented by ChatGPT.
"""

import random
import sys
from copy import deepcopy
import pygame
import numpy as np
from neural_network import NeuralNetwork

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BIRD_WIDTH = 50
BIRD_HEIGHT = 40
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
GRAVITY = 800
BIRD_JUMP = -360
PIPE_GAP = 150
PIPE_VELOCITY = -250
BIRD_MAX_VELOCITY = 1000

# Colors
WHITE = (255, 255, 255)
TRANSPARENT_WHITE = (255, 255, 255, 150)
BLACK = (0, 0, 0)

# Neuroevolution
N_BIRDS = 800  # ! Preferably an even number
NN_ARCHITECTURE = (4, 6, 6, 1)
N_LAYERS = len(NN_ARCHITECTURE)
ACTIVATION_FUNCTION = 'sigmoid'
MUTATION_RATE = 0.1
MUTATION_VALUE = 1

# OPTIMISATION AND TIMING
MAX_FPS = 60
SPEED_UP_DT = 0.02

# Create the game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Load images and create masks
yellow_bird_img = pygame.image.load("assets/bird.png")
yellow_bird_img = pygame.transform.scale(yellow_bird_img, (BIRD_WIDTH, BIRD_HEIGHT))

red_bird_img = pygame.image.load("assets/red_bird.png")
red_bird_img = pygame.transform.scale(red_bird_img, (BIRD_WIDTH, BIRD_HEIGHT))

blue_bird_img = pygame.image.load("assets/blue_bird.png")
blue_bird_img = pygame.transform.scale(blue_bird_img, (BIRD_WIDTH, BIRD_HEIGHT))

green_bird_img = pygame.image.load("assets/green_bird.png")
green_bird_img = pygame.transform.scale(green_bird_img, (BIRD_WIDTH, BIRD_HEIGHT))

pipe_img = pygame.image.load("assets/pipe.png")
pipe_img = pygame.transform.scale(pipe_img, (PIPE_WIDTH, PIPE_HEIGHT))

upper_pipe_img = pygame.transform.flip(pipe_img, False, True)

bg_img = pygame.image.load("assets/bg.png")
bg_img = pygame.transform.scale(bg_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

# Neural Network Representation Minimap
NN_REPR_WIDTH = 100
NN_REPR_HEIGHT = 100
NODE_SIDE_LENGHT = 6


class Bird:
    """
    Represents a single bird in the Flappy Bird game.

    Attributes:
        x (int): The x-coordinate of the bird.
        y (int): The y-coordinate of the bird.
        velocity (float): The vertical velocity of the bird.
        is_alive (bool): A flag indicating whether the bird is alive.
        brain (NeuralNetwork): The neural network controlling the bird.
        color (str): The color of the bird ('yellow', 'red', 'blue', 'green').
        score (float): The score of the bird, used for neuroevolution.
    """

    def __init__(self, color='yellow'):
        """
        Initializes the Bird object.

        Args:
            color (str): The color of the bird ('yellow', 'red', 'blue', 'green').
                         Defaults to 'yellow'.
        """
        self.brain = NeuralNetwork(NN_ARCHITECTURE, ACTIVATION_FUNCTION)
        self.color = color
        self.score = 0
        self.velocity = 0
        self.reset()

    def jump(self):
        """
        Makes the bird jump by setting its velocity to the jump value.
        """
        if self.is_alive:
            self.velocity = BIRD_JUMP

    def move(self, dt):
        """
        Updates the bird's position based on its velocity and the time delta.

        Args:
            dt (float): The time delta since the last update.
        """
        if self.is_alive:
            self.velocity += GRAVITY * dt
            self.velocity = min(self.velocity, BIRD_MAX_VELOCITY)
            self.y += self.velocity * dt
            self.score += self.velocity * dt

    def draw(self):
        """
        Draws the bird on the screen based on its color and position.
        """
        if self.is_alive:
            screen.blit(
                red_bird_img if self.color == 'red' else
                blue_bird_img if self.color == 'blue' else
                green_bird_img if self.color == 'green' else
                yellow_bird_img,
                (self.x, self.y))

    def should_jump(self, closest_next_pipe: 'Pipe'):
        """
        Determines whether the bird should jump based on its neural network's output.

        Args:
            closest_next_pipe (Pipe): The closest next pipe to the bird.

        Returns:
            bool: True if the bird should jump, False otherwise.
        """
        input1 = self.y / SCREEN_HEIGHT
        input2 = self.velocity / BIRD_MAX_VELOCITY
        input3 = closest_next_pipe.x / SCREEN_WIDTH
        input3 = input3 if input3 < 1 else 1
        input4 = closest_next_pipe.height / 400
        input_matrix = np.array([input1, input2, input3, input4])
        output = self.brain.forward(input_matrix)[-1]
        return output[0] > 0.5

    def reset(self):
        """
        Resets the bird's position, velocity, and status.
        """
        self.x = 50
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.is_alive = True


class Pipe:
    """
    Represents a pipe obstacle in the Flappy Bird game.

    Attributes:
        x (int): The x-coordinate of the pipe.
        height (int): The height of the lower pipe.
        is_passed (bool): A flag indicating whether the pipe has been passed by a bird.
    """

    def __init__(self, x):
        """
        Initializes the Pipe object.

        Args:
            x (int): The initial x-coordinate of the pipe.
        """
        self.x = x
        self.height = random.randint(80, 370)
        self.is_passed = False

    def move(self, dt):
        """
        Updates the pipe's position based on the time delta.

        Args:
            dt (float): The time delta since the last update.
        """
        self.x += PIPE_VELOCITY * dt

    def draw(self):
        """
        Draws the pipe on the screen at its current position.
        """
        screen.blit(upper_pipe_img, (self.x, self.height - PIPE_HEIGHT))
        screen.blit(pipe_img, (self.x, self.height + PIPE_GAP))


def check_collision(bird, pipes):
    """
    Checks for collisions between a bird and the pipes.

    Args:
        bird (Bird): The bird object to check for collisions.
        pipes (list of Pipe): The list of pipe objects to check against.

    Returns:
        bool: True if the bird collides with any pipe, False otherwise.
    """
    for pipe in pipes:
        if pipe.x <= bird.x + BIRD_WIDTH and pipe.x + PIPE_WIDTH >= bird.x:
            if bird.y <= pipe.height or bird.y + BIRD_HEIGHT >= pipe.height + PIPE_GAP:
                return True
    return False


def generate_nn_represetation():
    """
    Generates a visual representation of the neural network used by the birds.

    This function creates a surface displaying the structure of the neural network,
    showing the nodes and connections.

    Returns:
        pygame.Surface: A surface containing the neural network representation.
    """
    nn_repr = pygame.Surface((NN_REPR_WIDTH, NN_REPR_HEIGHT), pygame.SRCALPHA)

    nodes = []
    for i in range(N_LAYERS):
        nodes.append([])
        a = (i * (NN_REPR_WIDTH - NODE_SIDE_LENGHT)/(N_LAYERS - 1))
        for j in range(NN_ARCHITECTURE[i]):
            b = ((j+1)*NN_REPR_HEIGHT - (NN_ARCHITECTURE[i]-j)
                 * NODE_SIDE_LENGHT) / (NN_ARCHITECTURE[i]+1)
            rect = pygame.Rect(a, b, NODE_SIDE_LENGHT, NODE_SIDE_LENGHT)
            nodes[-1].append(rect)

    for i in range(N_LAYERS-1):
        for a in nodes[i]:
            for b in nodes[i+1]:
                pygame.draw.line(nn_repr, TRANSPARENT_WHITE, a.center, b.center, 1)

    for rect_layer in nodes:
        for rect in rect_layer:
            pygame.draw.rect(nn_repr, WHITE, rect)

    return nn_repr


def main():
    """
    The main function that runs the Flappy Bird game loop and handles the neuroevolution process.

    This function initializes the game environment, handles user input, updates the game state, and 
    manages the neuroevolution of the birds across generations. The game loop continues indefinitely 
    until the user quits the game.

    Key features of this function include:
    - Initializing the game environment and setting up the birds and pipes.
    - Handling user input to control game speed, print FPS, or display neural network information.
    - Updating the positions of the birds and pipes, checking for collisions, and managing the game.
    - Performing neuroevolution by selecting the best-performing bird and creating new generations 
      through mutation.

    The function also includes an optional "ultimate speed-up" mode that accelerates the game 
    for faster neuroevolution.

    The game loop performs the following tasks:
    1. Handles user input to control game speed, display FPS, or view neural network information.
    2. Updates the game state by moving pipes and birds, checking for collisions.
    3. Handles the game logic for when all birds in the current generation die:
       - Displays the results of the round.
       - Creates new generations of birds by duplicating the best-performing ones and applying
         random mutations to their neural network (brain).
       - Resets the game environment for the new generation.
    4. Displays game statistics, including generation number, number of birds alive, and scores.
    5. Adjusts the frame rate based on the game speed and ultimate speed-up mode.

    The function ensures that the game runs at a smooth frame rate and updates the game efficiently.
    """
    clock = pygame.time.Clock()
    birds = [Bird() for _ in range(N_BIRDS)]
    generation = 1
    dead_birds = []
    alive_birds = birds.copy()
    pipes = [Pipe(SCREEN_WIDTH + 1000)]
    game_speed = 1
    ultimate_speed_up = False

    score = 0
    best_score = 0
    best_generation = 1
    font = pygame.font.SysFont('Arial', 20)

    nn_repr = generate_nn_represetation()

    dt = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    print(clock.get_fps())
                if event.key == pygame.K_RETURN:
                    alive_birds[0].brain.print_info()
                if event.key == pygame.K_RIGHT:
                    game_speed += 1
                    print(f"The game's speed is now {game_speed}")
                if event.key == pygame.K_LEFT:
                    if game_speed > 1:
                        game_speed -= 1
                    print(f"The game's speed is now {game_speed}")
                if event.key == pygame.K_s:
                    ultimate_speed_up = not ultimate_speed_up

        if not ultimate_speed_up:
            screen.blit(bg_img, (0, 0))

        new_pipes = []
        for pipe in pipes:
            pipe.move(dt)
            if pipe.x + PIPE_WIDTH < 0:
                # Delete the old Pipe and replace it with a new one
                new_pipes.append(Pipe(SCREEN_WIDTH))
            else:
                new_pipes.append(pipe)  # Keep the same pipe
            if pipes[0].x < 0 and not pipes[0].is_passed:
                pipes[0].is_passed = True
                score += 1
                if score > best_score:
                    best_score = score
                    best_generation = generation
            if not ultimate_speed_up:
                pipe.draw()
        pipes = new_pipes

        for bird in alive_birds:
            if bird.should_jump(pipes[0]):
                bird.jump()
            bird.move(dt)
            if check_collision(bird, pipes) or (bird.y > SCREEN_HEIGHT or bird.y < 0):
                bird.is_alive = False
                dead_birds.append(bird)
                alive_birds.remove(bird)
            if not ultimate_speed_up:
                bird.draw()

        if len(alive_birds) == 0:
            # Printing results
            best_bird = dead_birds[-1]
            print()
            print(f'Round completed! (Generation {generation})')
            print(f'Best score: {score}')
            print('Best bird network:')
            best_bird.brain.print_info()

            # New generation creation
            best_bird.reset()
            best_bird.color = 'blue'
            birds.clear()
            for _ in range(N_BIRDS // 2):
                birds.append(Bird())
            for _ in range(N_BIRDS // 2):
                new_bird = deepcopy(best_bird)
                new_bird.brain.mutate(MUTATION_RATE, MUTATION_VALUE)
                new_bird.color = 'red'
                birds.append(new_bird)
            birds.append(best_bird)
            dead_birds.clear()
            alive_birds = birds.copy()

            # Game reset
            generation += 1
            pipes = [Pipe(SCREEN_WIDTH + 1000)]
            score = 0

        if not ultimate_speed_up:
            # Displaying game statistics
            screen.blit(font.render("Generation: " + str(generation), True, BLACK), (10, 10))
            screen.blit(font.render("Birds Alive: " + str(len(alive_birds)), True, BLACK), (10, 30))
            screen.blit(font.render("Score: " + str(score), True, BLACK), (10, 50))
            screen.blit(
                font.render(f"Best Score: {best_score} by Gen{best_generation}", True, BLACK),
                (10, 70))

            screen.blit(nn_repr, (SCREEN_WIDTH - NN_REPR_HEIGHT - 10, 5))

            pygame.display.update()

        if not ultimate_speed_up:
            tick = clock.tick(MAX_FPS)
            if tick > 50:
                tick = 20  # FREEZE HANDLING
            dt = game_speed * tick / 1000
        else:
            dt = SPEED_UP_DT


if __name__ == "__main__":
    main()
