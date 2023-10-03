import gymnasium as gym
import numpy as np
import cv2
import neat
import pickle


env = gym.make("ALE/MsPacman-v5", render_mode = "human")

def eval_genomes(genomes, config):

    for genome_id, genome in genome:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)


    
config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)

