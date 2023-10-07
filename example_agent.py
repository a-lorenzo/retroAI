import gymnasium as gym
import numpy as np
import cv2
import neat
import pickle


env = gym.make("ALE/MsPacman-v5", render_mode="human")
imgarray = []

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        ob = np.array(env.reset())
        ac = env.action_space.sample()
        
        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)
        print(inx, iny, inc)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False


        while not done:
            env.render()
            frame += 1

            # Reducing screenshot of emulator
            #print("Shape of ob:", ob.shape if ob is not None else None)
            #print("Data type of ob:", ob.dtype if ob is not None else None)

            #ob = cv2.resize(ob, (inx, iny))
            #ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            #ob = np.reshape(ob, (inx, iny))

            imgarray.clear()

            # Flattening image into 1D array
            for x in ob:
                for y in x:
                    if not isinstance(y[0][0], str):
                        imgarray.append(float(y[0][0]) / 255.0)


            nnOutput = net.activate(imgarray)
            print("\n\n", "NEURALNET OUTPUT:\n=============")
            print(nnOutput)
            
            action = np.argmax(nnOutput)
            result  = env.step(action)
            observation, reward, terminated, truncated, info = env.step(action)

            

            

            
config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)
winner = p.run(eval_genomes)
