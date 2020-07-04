from dqn_env import TrainLine
from RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as mplt
import tensorflow as tf
import pandas as pd

def plot(r,ylabel):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(r)), r, linewidth=1)
    plt.ylabel(ylabel)
    plt.xlabel('training episodes')
    plt.show()

def draw_mean_reward(r):
    import matplotlib.pyplot as plt
    x_10 = []
    temp = []
    count = 0
    for i in range (len(r)):
        temp.append(r[i])
        count += 1
        if count >= 10:
            x_10.append(sum(temp) / 10)
            temp = []
            count = 0
    plt.plot(np.arange(len(x_10)), x_10, linewidth=1)
    plt.ylabel('mean_reward')
    plt.xlabel('training episodes X10')
    plt.show()

def run_train():
    total_step = 0
    Max_iteras= 5000
    for episode in range(Max_iteras):
        #训练5000次
        r1_max = 0
        step = 0
        r1 = 0
        pl=[]
        vl=[]
        # initial observation
        observation = env.reset()
        #env.bef_print()
        while True:
            # fresh env
            #env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)
            ob = observation[2]
            if 500 > episode > 100 and step % 10 == 0:
                if env.u > 0 and ob > 0:
                    action = 4
                elif env.u < 0 and ob > 0:
                    action = 6
                elif env.u > 0 and ob < 0:
                    action = 0
                elif env.u < 0 and ob < 0:
                    action = 2
                else:
                    action = action
            if 3000 > episode >= 500 and step % 4 == 0:
                if env.u > 0 and ob > 0:
                    action = 4
                elif env.u < 0 and ob > 0:
                    action = 6
                elif env.u > 0 and ob < 0:
                    action = 0
                elif env.u < 0 and ob < 0:
                    action = 2
                else:
                    action = action
            if 4500 > episode >= 3000 and step % 2 == 0:
                if env.u > 0 and ob > 0:
                    action = 4
                elif env.u < 0 and ob > 0:
                    action = 6
                elif env.u > 0 and ob < 0:
                    action = 0
                elif env.u < 0 and ob < 0:
                    action = 2
                else:
                    action = action
            if 5000 > episode >= 4500 and step % 2 == 0:
                if env.u > 0 and ob > 0:
                    action = 4
                elif env.u < 0 and ob > 0:
                    action = 6
                elif env.u > 0 and ob < 0:
                    action = 0
                elif env.u < 0 and ob < 0:
                    action = 2
                else:
                    action = action

            # RL take action and get next observation and reward
            observation_,E,reward, done,  _ = env.step(action) # action =0-6 最后会被转换到转化为[-0.3, 0.3]

            r1 += reward/10

            RL.store_transition(observation, action, reward, observation_)
            if (total_step > 10000 and total_step % 5 == 0 ):
                RL.learn()
            # swap observation
            observation = observation_
            o1 =observation
            if episode%20==0 or episode==Max_iteras-1:
                pl.append(observation[0])
                vl.append(observation[1])
            # break while loop when end of this episode
            if done:
                r.append(r1)
                energy.append(E)
                tlist.append(observation_[2])
                #曲线判定函数，决定是否保存曲线 ：旅行距离是否合适，时间是否接近，以episode_speed.csv为 文件名
                if r1 > r1_max and episode>1500 and episode%20==0:
                    r1_max =r1
                    Curve=np.mat([pl,vl])
                    CurveData=pd.DataFrame(data=Curve.T,columns=['s','v'])
                    CurveData.to_csv("./Curve/"+str(episode)+"_CurveData.csv")                    
                if episode==Max_iteras-1:
                    print(r1)
                    f1 = open('datat.txt', 'r+')
                    f1.read()
                    print(episode, (step + 5)/5, file=f1)
                    f1.close()
                    r.append(r1)
                    print('Episode finished after {} timesteps'.format((step + 5)/5))
                break
#            if (5000 > episode >= 4500):
#                 print(o1)
#                 f2 = open('vs.txt', 'r+')
#                 f2.close()
#                 break
            step += 1
            total_step += 1
        #最后打印结果
        print(episode)
        if episode%20==0 or episode==Max_iteras-1:
            mplt.plot(pl,vl)
            mplt.savefig("./img/"+str(episode)+"v-s.png")
            mplt.show()            
    return
            
# end of game


if __name__ == "__main__":
    global r,energy,tlist,RL
    tf.reset_default_graph()
    env = TrainLine()
    env.seed(1)
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.001,
                      reward_decay=0.95,
                      e_greedy=0.9,
                      replace_target_iter=400,
                      memory_size=10000,
                      e_greedy_increment=None,
                      # output_graph=True
                      )

    energy = []
    r = []
    tlist = [] 
    run_train()
    RL.plot_cost()
    plot(r,'reward')
    plot(energy,'energy')
    plot(tlist,'time')
    draw_mean_reward(r)
    rdata = pd.DataFrame(r)
    rdata.to_csv("reward.csv")
    tdata = pd.DataFrame(tlist)
    tdata.to_csv("timeError.csv")
    costData = pd.DataFrame(RL.cost_his)
    costData.to_csv("costData.csv")
    Edata = pd.DataFrame(energy)
    Edata.to_csv("EData.csv")