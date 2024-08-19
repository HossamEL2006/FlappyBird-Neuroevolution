# FlappyBird-Neuroevolution

A Flappy Bird clone implemented in Python using Pygame, featuring a neuroevolution algorithm to train birds to play the game using neural networks. This project combines game development with machine learning concepts to create an engaging simulation of neural network training.

## Features

- **Flappy Bird Game**: Playable version of the classic Flappy Bird game with smooth graphics and physics.
- **Neuroevolution Algorithm**: Uses a simple neuroevolution algorithm to train multiple birds (agents) to navigate through pipes.
- **Neural Network Integration**: Each bird is controlled by a neural network that evolves over generations to improve performance.
- **Visualization**: Displays neural network architecture and game statistics in real-time.

## Usage
To start the game, run the following command in your terminal:
```bash
python main.py
```

The game will open in a new window. You can control the following features using keyboard inputs:

- **F**: Print the current frame rate (FPS).
- **ENTER**: Print the neural network information of the first bird.
- **RIGHT** Arrow: Increase the game speed.
- **LEFT** Arrow: Decrease the game speed (minimum speed is 1).
- **S**: Toggle "ultimate speed-up" mode, which accelerates the game.

## Game Mechanics
- **Birds**: Controlled by neural networks that evolve over time. Birds will attempt to navigate through pipes by jumping at appropriate times.
- **Pipes**: Obstacles that move from right to left. The gap between pipes allows birds to pass through.
- **Neuroevolution**: Birds that survive longer and score higher contribute to the next generation. The best-performing bird's neural network is used as a basis for new generations, with some mutations.

## Contributing
Contributions are welcome! If you have any improvements, bug fixes, or new features, please feel free to open an issue or submit a pull request.
- Fork the repository
- Create a new branch for your feature or fix
- Commit your changes
- Push to the branch
- Create a pull request with a detailed description of your changes

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or feedback, please contact me at:
- **Email**: hoss.lehhit@gmail.com
- **GitHub**: HossamEL2006


### Enjoy the game and happy coding!
