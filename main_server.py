import os
import random
import sys
from collections import deque

import gym
from gym.wrappers import Monitor
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input,
                                     MaxPooling2D, ReLU, Softmax)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Nadam

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

transition = np.dtype(
    [
        ('state', np.float64, (96, 96, 3)), 
        ('action', np.float64, (3,)),
        ('reward', np.float64), 
        ('next_state', np.float64, (96, 96, 3)),
        ('done', np.bool_)
    ]
)
MAX_SIZE = 500
class myCarRacing:
    
    def __init__(self, gameName):
        self.env = wrap_env(gym.make(gameName))

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_model()

        # 經驗庫
        self.memory_buffer = np.empty(MAX_SIZE, dtype=transition)
        self.counter = 0
        # Q_value的discount rate，以便計算未來reward的折扣回報
        self.gamma = 0.95
        # 貪婪選擇法的隨機選擇行為的程度
        self.epsilon = 1.0
        # 上述參數的衰減率
        self.epsilon_decay = 0.995
        # 最小隨機探索的概率
        self.epsilon_min = 0.01

    def build_model(self):
        # 自動增長 GPU 記憶體用量
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

        # 設定 Keras 使用的 Session
        tf.compat.v1.keras.backend.set_session(sess)

        # model
        inputLayer = Input(shape=self.env.observation_space.shape)

        convLayer = Conv2D(8, 4, strides=2, padding="valid")(inputLayer)
        activationLayer = ReLU()(convLayer)
        convLayer = Conv2D(16, 3, strides=2, padding="valid")(activationLayer)
        activationLayer = ReLU()(convLayer)
        convLayer = Conv2D(32, 3, strides=2, padding="valid")(activationLayer)
        activationLayer = ReLU()(convLayer)
        convLayer = Conv2D(64, 3, strides=2, padding="valid")(activationLayer)
        activationLayer = ReLU()(convLayer)
        convLayer = Conv2D(128, 3, strides=1, padding="valid")(activationLayer)
        activationLayer = ReLU()(convLayer)
        convLayer = Conv2D(256, 3, strides=1, padding="valid")(activationLayer)
        activationLayer = ReLU()(convLayer)

        flattenLayer = Flatten()(activationLayer)

        denseLayer = Dense(100)(flattenLayer)
        denseLayer = ReLU()(denseLayer)

        denseLayer = Dense(self.env.action_space.shape[0], activation='softmax')(denseLayer)

        model = Model(inputs=inputLayer, outputs=denseLayer)
        model.summary()
        return model

    def update_target_model(self):
        """更新target_model
        """
        self.target_model.set_weights(self.model.get_weights())

    def egreedy_action(self, state):
        """ε-greedy選擇action

        Arguments:
            state: 狀態

        Returns:
            action: 動作
        """
        if np.random.rand() <= self.epsilon:
            return [random.random()*2-1 for i in range(3)]
        else:
            q_values = self.model.predict(state)[0]
            return q_values
    
    def remember(self, state, action, reward, next_state, done):
        """向經驗池添加數據
        Arguments:
            state: 狀態
            action: 動作
            reward: 回報
            next_state: 下一個狀態
            done: 遊戲結束標誌
        """
        self.memory_buffer[self.counter] = (state, action, reward, next_state, done)
        self.counter+=1
        if(self.counter == MAX_SIZE):
            self.counter = 0
            return True
        else:
            return False
        # item = [state, action, reward, next_state, done]
        # self.memory_buffer.append(item)

    def update_epsilon(self):
        """更新epsilon
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch):
        """batch數據處理
        Arguments:
            batch: batch size
        Returns:
            X: states
            y: [Q_value1, Q_value2]
        """
         # 從經驗池中隨機採樣一個batch
        # randomList = random.sample(np.linspace(0, batch-1, batch), batch)
        data = self.memory_buffer[random.sample(list(np.linspace(0, batch-1, batch, dtype=np.int32)), batch)]
        '''
        data = [[state, action, reward, next_state, done],
                [state, action, reward, next_state, done]...]
        '''
        # 生成Q_target。
        states = np.array([d["state"] for d in data])
        next_states = np.array([d["next_state"] for d in data])

        y = self.model.predict(states)
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            
            if not done:
                for j in range(3):
                    y[i][j] = target + self.gamma * q[i][j]
                
        return states, y

    def train(self, episode, batch):
        """訓練
        Arguments:
            episode:遊戲次數
            batch:batch size
        Returns:
            history:訓練紀錄
        """
        self.model.compile(loss=BinaryCrossentropy(), optimizer=Nadam(learning_rate=0.001))

        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        count = 0

        observation = self.env.reset()
        
        for i in range(episode):
            observation = self.env.reset()
            
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                self.env.render()
                x = observation.reshape([-1] + list(self.env.observation_space.shape))
                # 貪心演算法
                action = self.egreedy_action(x)
                # print(action)
                observation, reward, done, _ = self.env.step(action)

                reward_sum += reward
                # self.remember(x[0], action, reward, observation, done)

                if(self.remember(x[0], action, reward, observation, done)):
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    #減少貪心程度
                    self.update_epsilon()

                    #更新target model
                    if count != 0 and count % 20 == 0:
                        self.update_target_model()

            if i % 1 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)
                self.model.save("tensorflow.h5")
                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss, self.epsilon))
        
        self.model.save_weights('model.h5')
        return history

    def play(self):
        observation = self.env.reset()

        count = 0
        reward_sum = 0
        random_episodes = 0

        while random_episodes < 10:
            self.env.render()

            x = observation.reshape([-1] + list(self.env.observation_space.shape))
            q_values = self.model.predict(x)
            action = q_values["action"]
            observation, reward, done, _ = self.env.step(action)

            count += 1
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                observation = self.env.reset()

        self.env.close()

if __name__ == "__main__":
    game = myCarRacing("CarRacing-v0")
    try:
        from tensorflow.keras.models import load_model
        game.model = load_model("tensorflow.h5") 
        print("load model.")
    except:
        pass
    history = game.train(100, 16)
    game.play()
