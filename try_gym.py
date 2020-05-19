import gym
import numpy as np
import sys

def try_gym():
    # 使用gym创建一个CartPole环境
    # 这个环境可以接收一个action，返回执行action后的观测值，奖励与游戏是否结束
    env = gym.make("CarRacing-v0") 
    # env = gym.make("CartPole-v0") 

    # print("action",env.action_space)
    # print("observation", env.observation_space.shape)
    # for i in range(10):
    #     print(env.action_space.sample())
    # sys.exit()

    # 重置游戏环境
    observation = env.reset()
    # print(observation.reshape([-1] + list(env.observation_space.shape)))
    # sys.exit()
    # 游戏轮数
    random_episodes = 0
    # 每轮游戏的Reward总和
    reward_sum = 0
    count = 0
    while random_episodes < 10:
        # 渲染显示游戏效果
        env.render()
        # 随机生成一个action，即向左移动或者向右移动。
        # 然后接收执行action之后的反馈值
        observation, reward, done, _ = env.step(env.action_space.sample())
        reward_sum += reward
        count += 1
        # 如果游戏结束，打印Reward总和，重置游戏
        if done:
            random_episodes += 1
            print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
            reward_sum = 0
            count = 0
            env.reset()


if __name__ == '__main__':
    try_gym()