
import numpy as np
import math
import gym
from gym import spaces
from gym.utils import seeding
from interval import Interval
import matplotlib.pyplot as plt

class TrainLine(gym.Env):
import numpy as np
import math
import gym
from gym import spaces
from gym.utils import seeding
from interval import Interval
import matplotlib.pyplot as plt

UNIT_H = 20   # pixels
SPEED_H = 25  # grid height
DISTANCE_W =1299  # grid width
UNIT_W=1
class TrainLine(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    def __init__(self):

        self.action_space = [-0.3, 0.3]
        self.S = 1275
        self.max_speed = 80/3.6
        self.T = 100          #运行总时间
        self.ac = 0.8
        self.de = 1
        self.resistance = 0.1
        self.n_features = 3
        self.t = 0.2          #运行时间步长
        self.low = np.array([0, 0])
        self.high = np.array([self.S, self.max_speed])
        self.a = 1
        self.viewer = None
        self.observation_space = spaces.Box(self.low, self.high)
        self.action_space = Interval(-0.3, 0.3)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        position1, velocity1 = self.state1
        u = np.round((self.u + action), 1) #action是加速度的增量
        u = max(-0.8, u)
        u = min(0.8, u)
        self.u = u
        pos_cha = max(velocity1 * self.t + 0.5 * u * (self.t) ** 2, 0)  # 位置增量
        position1 += pos_cha
        velocity1 += self.t * u
        if (velocity1 > self._get_max_speed(position1)): velocity1 = self._get_max_speed(position1)
        if (velocity1 < 0): velocity1 = 0
        if u >= 0:
           reward = -(u + self.resistance) * pos_cha * 0.2
        else:
           reward = 0
        self.state1 = (position1, velocity1)
        position = position1
        velocity = velocity1
        if position1 >= 1274.5:
            done = True
            reward += -0.3*(abs(self.step1 * self.t - self.T)) ** 2
        elif self.step1*self.t >= 300:
            done = True
        else:
            done = False
        if position1 <= 1274.5:
            vr = 1 / (self.get_refer_time(position1) - self.get_refer_time(position1-1))
        else:
            vr = 0
        v_dif = vr - velocity1
        reward -= abs(v_dif)**2 * 0.3
        self.state1 = (position1, velocity1)
        self.state = (position, velocity, v_dif) #位置、速度、加速度,能耗
        self.step1 += 1
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.EC=0   #消耗的能耗
        self.step1 = 2
        self.u = 0.3 #初始加速度为0.3m/s2
        self.state = np.array([1.6, 1.6, 0])
        self.state1 = np.array([1.6, 1.6])
        return np.array(self.state)

    def _get_max_speed(self, position): #全长1275米,限速80km/h
        s1 = (80/3.6)**2/2/self.ac
        s2 = 1275-((80/3.6)**2/2/self.de)
        if position < s1:
            self.max_speed = math.sqrt(2*self.ac*position)
        elif position < s2:
            self.max_speed = 80/3.6
        elif position < 1275:
            self.max_speed = math.sqrt((1275-position)*2*self.de)
        else:
            self.max_speed = 0
        return self.max_speed

    def get_refer_time(self, position):
        s1 = (80 / 3.6) ** 2 / 2 / self.ac
        s2 = 1275 - ((80 / 3.6) ** 2 / 2 / self.de)
        t1 = 80/3.6/self.ac
        t2 = (s2 - s1) / (80/3.6)
        t3 = 80/3.6/self.de
        tz = t1 + t2 + t3
        if position <= s1:  #从s=10,v=4 开始
            v_max = math.sqrt((position )*2*self.ac )
            t_min = (v_max)/self.ac
        elif position <= s2:
            t_min = (position - s1) / (80/3.6)
            t_min = t1 + t_min
        elif position <= 1275:
            v_max = math.sqrt((80/3.6)**2 - (position - s2)*2*self.de)
            t_min = (80/3.6 - v_max)/self.de
            t_min = t1 + t2 + t_min
        else:
            t_min = tz
        tr = (t_min / tz) * self.T
        self.tr = tr
        return self.tr

    def bef_print(self):
        for i in range(2):
            position = 0.5 * self.ac * i**2
            velocity = i * self.ac
            f1 = open('C:\chengxu\datat.txt', 'r+')
            f1.read()
            print(position, velocity, file=f1)
            f1.close()

    def render(self, mode = 'human'):
            screen_width = 1300
            screen_height = 500

            world_width = 2400
            world_height=25
            scale_w = screen_width / world_width
            scale_h=screen_height/world_height
            trainwidth = 60
            trainheight = 20

            self.store=  [0 for x in range(0, 1000)]
            self.store_= [0 for x in range(0, 1000)]

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(screen_width, screen_height)
                xs = np.linspace(0, 2400, 2400)
                ys = np.linspace(0,25,25)
                xys = list(zip(xs*scale_w, ys * scale_h))

                self.track = rendering.make_polyline(xys)
                self.track.set_linewidth(8)
                #self.viewer.add_geom(self.track)

                clearance = 10

                l, r, t, b = -trainwidth / 2, trainwidth / 2, trainheight, 0
                train = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                train.add_attr(rendering.Transform(translation=(0, clearance)))
                self.traintrans = rendering.Transform()
                train.add_attr(self.traintrans)
                self.viewer.add_geom(train)
                frontwheel = rendering.make_circle(trainheight / 2.5)
                frontwheel.set_color(.5, .5, .5)
                frontwheel.add_attr(rendering.Transform(translation=(trainwidth / 4, clearance)))
                frontwheel.add_attr(self.traintrans)
                self.viewer.add_geom(frontwheel)
                backwheel = rendering.make_circle(trainheight / 2.5)
                backwheel.add_attr(rendering.Transform(translation=(-trainwidth / 4, clearance)))
                backwheel.add_attr(self.traintrans)
                backwheel.set_color(.5, .5, .5)
                self.viewer.add_geom(backwheel)
                flagx = 2350 * scale_w
                flagy1 = 0
                flagy2 = flagy1 + 50
                flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
                self.viewer.add_geom(flagpole)
                flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
                flag.set_color(.8, .8, 0)
                self.viewer.add_geom(flag)


                x1=rendering.Line((0,60/3.6*scale_h),(50*scale_w,60/3.6*scale_h))
                self.viewer.add_geom(x1)
                x1.set_color(255,0,0)
                x2=rendering.Line((50*scale_w,60/3.6*scale_h),(50*scale_w,70/3.6*scale_h))
                x2.set_color(255, 0, 0)
                self.viewer.add_geom(x2)
                x3=rendering.Line((50*scale_w,70/3.6*scale_h),(1000*scale_w,70/3.6*scale_h))
                x3.set_color(255, 0, 0)
                self.viewer.add_geom(x3)
                x4=rendering.Line((1000*scale_w,70/3.6*scale_h),(1000*scale_w,80/3.6*scale_h))
                x4.set_color(255, 0, 0)
                self.viewer.add_geom(x4)
                x5=rendering.Line((1000*scale_w,80/3.6*scale_h),(2200*scale_w,80/3.6*scale_h))
                x5.set_color(255, 0, 0)
                self.viewer.add_geom(x5)
                x6=rendering.Line((2200*scale_w,80/3.6*scale_h),(2200*scale_w,60/3.6*scale_h))
                x6.set_color(255, 0, 0)
                self.viewer.add_geom(x6)
                x7=rendering.Line((2200*scale_w,60/3.6*scale_h),(2400*scale_w,60/3.6*scale_h))
                x7.set_color(255, 0, 0)
                self.viewer.add_geom(x7)
                if self.state[0]==0:
                   x0 = 0
                   y0 = 0
                   x1 =100
                   y1 = 20
                   outline = rendering.Line((x0*scale_w, y0*scale_h), (x1*scale_w, y1*scale_h))
                   outline.set_color(0,255,0)
                   self.viewer.add_geom(outline)



            pos = self.state[0]
            self.traintrans.set_translation(pos*scale_w, 0)
            #self.traintrans.set_rotation(math.cos(3 * pos))



            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
           if self.viewer: self.viewer.close()
