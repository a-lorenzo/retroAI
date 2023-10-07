import gymnasium as gym
import numpy as np
import cv2
import neat
import pickle
import visualize

env = gym.make("ALE/MsPacman-v5", render_mode="human")
env.metadata['render_fps'] = 120
imgarray = []

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        ob = np.array(env.reset(), dtype = object)
        ac = env.action_space.sample()
        
        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        #net.visualize()
        #visualize.draw_net(config, net, True)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        score = 0
        score_max = 0

        done = False


        while not done:
            env.render()
            frame += 1

            # Reducing screenshot of emulator
            #scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            #ob = cv2.resize(ob, (inx, iny))
            #ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            #ob = np.reshape(ob, (inx, iny))

            # Flattening image into 1D array
            for x in ob:
                for y in x:
                    if not isinstance(y[0][0], str):
                        imgarray.append(float(y[0][0]) / 255.0)

            
            nnOutput = net.activate(imgarray)
            
            action = np.argmax(nnOutput)
            result  = env.step(action)
            observation, reward, terminated, truncated, info = env.step(action)
            imgarray.clear()
            fitness_current += reward

            # Evaluating fitness with in-game score
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250 or terminated or truncated:
                done = True
                print("Genome: ", genome_id, "Fitness: ", fitness_current)

            genome.fitness = fitness_current
            
            

            
config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     'config-feedforward')


p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

# Run the neat algorithm for 10 generations and write the winner
winner = p.run(eval_genomes, 10)
print('\nBest genome:\n{!s}'.format(winner))
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

visualize.draw_net(config, winner, True)
visualize.plot_stats(stats, ylog=False, view=True)
visualize.plot_species(stats, view=True)


