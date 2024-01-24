# Multi-Agent-Search
Exploring different multi-agent search algorithms, such as Minimax, Alpha-Beta Pruning, and Expectimax through Pac-Man


This project is written in Python

To run a classic game of Pac-Man, run command:
python pacman.py

To see the performance of Pac-Man when run with a Reflex Agent, run commands:
python pacman.py -p ReflexAgent -l testClassic
python pacman.py --frameTime 0 -p ReflexAgent -k 1
python pacman.py --frameTime 0 -p ReflexAgent -k 2

To see the performance of Pac-Man when run with a Minimax Agent, run command:
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4

To see Pac-Man purposely end the game ASAP when failure is unavoidable, run command:
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3

To see the performance of Pac-Man when run with Alpha-Beta pruning, run command:
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic

To see the performance of Pac-Man when run with Expectimax, run command:
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3

