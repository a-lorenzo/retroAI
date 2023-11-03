import gymnasium as gym
import numpy as np
import cv2
import neat
import pickle
import visualize
import matplotlib.pyplot as plt

env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
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
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        score = 0
        score_max = 0
        done = False


        while not done:
            img = env.render()
            frame += 1
            
            # Save the original image and display it for comparision; "human vision"
            # Can be commented out to increase speed
            originalimg = img
            originalimg = cv2.resize(originalimg, (300, 500), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Human vision", originalimg)
            cv2.waitKey(1)

            # Reducing screenshot of emulator
            img = cv2.resize(img, (inx, iny))
            # turn off color for nn vision
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
            
            # Reshaping seems to distort the image beyond recognition
            #img = np.reshape(img, (inx, iny))
            
            # Thresholding the background into black, everything else into gradient
            img = cv2.threshold(img, 57, 255, cv2.THRESH_TOZERO)[1]
            
            # Cropping unnecessary information from the image
            start_y = 0
            start_x = 0
            end_x = inx
            end_y = 16
            img = img[start_y:end_y, start_x:end_x]
            
            # Flattening image into 1D array
            imgarray = np.ndarray.flatten(img)             
            
            # Enlarge image without antialiasing to clearly see inputs to neural net; "computer vision"
            # Can be commented out to increase speed
            img = cv2.resize(img, (416, 240), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Neural Net Vision", img)
            cv2.waitKey(1)
            
            # Send inputs to neural net and get response to send back to environment
            nnOutput = net.activate(imgarray)
            action = np.argmax(nnOutput)
            result  = env.step(action)
            observation, reward, terminated, truncated, info = env.step(action)
            imgarray = []
            fitness_current += reward

            # Evaluating fitness with in-game score
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            # Terminate neural network when Ms. PacMan dies or when set time is up
            if done or counter == 250 or terminated or truncated:
                done = True
                print("Genome: ", genome_id, "Fitness: ", fitness_current)
            # Set genome's fitness
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

# Run the neat algorithm for n generations and write the winner
winner = p.run(eval_genomes, 100)
print('\nBest genome:\n{!s}'.format(winner))
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

visualize.plot_stats(stats, ylog=False, view=True)
visualize.plot_species(stats, view=True)


