import gym
import h5py
from keras.models import Sequential
from keras.models import  save_model
from keras.layers import Dense, Flatten
import numpy as np
import random

mapping = {
    0: [0, 0, 0, 1, 0, 0],
    1: [0, 0, 0, 0, 1, 0],
    2: [0, 0, 0, 1, 1, 0],
    3:[0,0,0,0,0,0]
}
observetime = 1
epsilon = 0.7
gamma = 0.9
mini_batch_size = 2000
train_time=1000000
env = gym.make('SuperMarioBros-1-1-Tiles-v0')
def Create_Model():
    neural_net_model = Sequential()
    neural_net_model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
    neural_net_model.add(Flatten())
    neural_net_model.add(Dense(18, init='uniform', activation='relu'))
    neural_net_model.add(Dense(10, init='uniform', activation='relu'))
    neural_net_model.add(Dense(4, init='uniform', activation='linear'))
    neural_net_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return neural_net_model

Train_data = []
neural_net_model=Create_Model()
neural_net_model.load_weights('/home/ntrex/Desktop/Mini project/model_8 500 _final.h5')
observation = env.reset()
obs = np.expand_dims(observation,axis=0)
state = np.stack((obs, obs), axis=1)
done = False
for t in range(observetime):
    env = gym.make('SuperMarioBros-1-1-Tiles-v0')
    observation = env.reset()
    obs = np.expand_dims(observation,axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False
    while done == False:
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 3, size=1)[0]
            action1 = mapping[action]
        else:
            Predicted_value = neural_net_model.predict(state)
            action = np.argmax(Predicted_value)
            action1 = mapping[action]
        observation_new, reward, done, info = env.step(action1)
        obs_new = np.expand_dims(observation_new, axis=0)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :],axis=1)
        state = state_new
    env.close()
    if t==0:
        arr=np.load('/home/ntrex/Desktop/Mini project/nishant1_good'+str(2)+'.npy')
        for i in arr:
            nd=[]
            ans=2
            for k,v in mapping.items():
                if v==i[1]:
                    ans=k
                    break
            i[1]=ans
            Train_data.append(i)
for t in range(train_time):
    Subset_training_data = random.sample(Train_data, mini_batch_size)

    inputs_shape = (mini_batch_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mini_batch_size, 4))

    for i in range(0, mini_batch_size):
        if i % 500 == 0:
            fp = open('./Final_Model/inp.txt', "a")
            fp.write(' model_' + str(t) + " "+str(i) + " _final1"+ '.h5'+" - ")
            neural_net_model.save('./Final_Model/model_' + str(t) + " "+str(i)+ " _final1"+ '.h5');
            env = gym.make('SuperMarioBros-1-1-Tiles-v0')
            observation = env.reset()
            obs = np.expand_dims(observation, axis=0)
            state = np.stack((obs, obs), axis=1)
            done = False
            tot_reward = 0.0
            while not done:
                env.render()
                Predicted_value = neural_net_model.predict(state)
                action = np.argmax(Predicted_value)
                if action==3:
                    action=2
                action1 = mapping[action]
                observation, reward, done, info = env.step(action1)
                obs = np.expand_dims(observation, axis=0)
                state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
                tot_reward += reward
            env.close()
            fp.write(str(tot_reward))
            fp.write('\n')
            fp.close()
            print(tot_reward)
        state = Subset_training_data[i][0]
        action = Subset_training_data[i][1]
        reward = Subset_training_data[i][2]
        state_new = Subset_training_data[i][3]
        done = Subset_training_data[i][4]
        inputs[i:i + 1] = np.expand_dims(state, axis=0)
        targets[i] = neural_net_model.predict(state)
        Next_target = neural_net_model.predict(state_new)
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Next_target)
        neural_net_model.train_on_batch(inputs, targets)
fp = open('inp.txt', "a")
env = gym.make('SuperMarioBros-1-1-Tiles-v0')
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
while not done:
    env.render()
    Predicted_value = neural_net_model.predict(state)
    action = np.argmax(Predicted_value)
    action1=mapping[action]
    observation, reward, done, info = env.step(action1)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
    tot_reward += reward
neural_net_model.save_weights('./try.h5')
env.close()
print('Game ended! Total reward: {}'.format(tot_reward))
fp.write('reward '+str(tot_reward))
fp.close()