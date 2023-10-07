# retroAI
A NEAT algorithm trained to play Ms. Pacman on the Atari 2600.
![Screenshot of the NEAT algorithm playing Ms. PacMan](https://i.imgur.com/aEMeYmZ.png)

## About
This is a personal project of my own attempt to train a NEAT algorithm to play a retro video game. The NEAT (Neuroevolution of augmenting topologies) algorithm is an evolutionary algorithm that creates generations of neural networks and evolves overtime based on the successful networks of previous generations.

In this case, the reward for the network is the direct score from the game itself; future generations of networks are created based on the highest scoring networks of previous generations.

The networks take a condensed array of pixel values as an input, and output a range of values from 1 - 9 that correspond to different directions Ms. PacMan can move. 

## Credits
* OpenAI's [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) was used to setup the Ms. PacMan environment.
* [neat-python](https://github.com/CodeReclaimers/neat-python) was used as the foundation for the NEAT algorithm.
* Lucas Thompson's [YouTube tutorial](https://www.youtube.com/playlist?list=PLTWFMbPFsvz3CeozHfeuJIXWAJMkPtAdS) was followed occasionally to familiarize myself with the similar [retro](https://github.com/openai/retro) library.