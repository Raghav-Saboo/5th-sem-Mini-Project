from keras.models import Sequential
from keras.models import  save_model
from keras.layers import Dense, Flatten

import numpy as np
import time
import gym
import h5py
env=gym.make('SuperMarioBros-1-1-Tiles-v0')
model = Sequential()
model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
model.add(Flatten())
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

mapping = {
    0: [0, 0, 0, 1, 0, 0],
    1: [0, 0, 0, 0, 1, 0],
    2: [0, 0, 0, 1, 1, 0],
    3:[0,0,0,0,0,0]
}
for i in range(0,4):
    env=gym.make('SuperMarioBros-1-1-Tiles-v0')
    arr=np.load('/home/ntrex/Desktop/nishant_b_'+str(i)+'.npy')
    env.reset()
    re=0
    for i in range(len(arr)):
        _,a,_,info=env.step(arr[i][1])
        time.sleep(0.01)
        re+=a
    print(re)
    #print(info)
    env.close()
for i in range(0,7):
    if i > 0:
        model.load_weights('/home/ntrex/Desktop/Mini project/selected/sel_' + str(i)+".h5")
    env = gym.make('SuperMarioBros-1-1-Tiles-v0')
    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False
    tot_reward = 0.0
    while not done:
        env.render()  # Uncomment to see game running
        time.sleep(0.01)
        Q = model.predict(state)
        action = np.argmax(Q)
        if action == 3:
            action = 2
        action1 = mapping[action]
        observation, reward, done, info = env.step(action1)
        obs = np.expand_dims(observation, axis=0)
        state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
        tot_reward += reward

    env.close()
    print(tot_reward)

for i in range(8,9):
    env=gym.make('SuperMarioBros-1-1-Tiles-v0')
    arr=np.load('/home/ntrex/Desktop/nishant'+str(i)+'.npy')
    env.reset()
    re=0
    for i in range(len(arr)):
        _,a,_,info=env.step(arr[i][1])
        time.sleep(0.01)
        re+=a
    print(re)
    env.close()
