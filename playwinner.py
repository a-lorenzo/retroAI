import gymnasium as gym
import numpy as np
import cv2
import neat
import pickle
import visualize
import matplotlib.pyplot as plt


env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
env.metadata['render_fps'] = 60
imgarray = []

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

with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

ob = env.reset()
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

    originalimg = img
    originalimg = cv2.resize(originalimg, (300, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Human vision", originalimg)
    cv2.waitKey(1)
                
    img = cv2.resize(img, (inx, iny))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 57, 255, cv2.THRESH_TOZERO)[1]
    start_y = 0
    start_x = 0
    end_x = inx
    end_y = 16
    img = img[start_y:end_y, start_x:end_x]
    imgarray = np.ndarray.flatten(img)
    img = cv2.resize(img, (416, 240), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Neural net vision", img)
    cv2.waitKey(1)

    nnOutput = net.activate(imgarray)
    action = np.argmax(nnOutput)
    result  = env.step(action)
    observation, reward, terminated, truncated, info = env.step(action)
    imgarray = []
    fitness_current += reward

    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
        counter = 0
    else:
        counter += 1
        
    if done or counter == 500 or terminated or truncated:
        done = True
        print("Fitness: ", fitness_current)
