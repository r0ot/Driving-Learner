#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Based on Chris Campbell's tutorial from iforce2d.net:
http://www.iforce2d.net/b2dtut/top-down-car
"""

from drive_framework import (Framework, Keys, main)
import math, numpy as np, random, neural_net
from Box2D import (b2Color, b2RayCastCallback, b2Vec2, )
import time, sys, datetime


class RayCastClosestCallback(b2RayCastCallback):
    """This callback finds the closest hit"""

    def __repr__(self):
        return 'Closest hit'

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction):
        '''
        Called for each fixture found in the query. You control how the ray
        proceeds by returning a float that indicates the fractional length of
        the ray. By returning 0, you set the ray length to zero. By returning
        the current fraction, you proceed to find the closest point. By
        returning 1, you continue with the original ray clipping. By returning
        -1, you will filter out the current fixture (the ray will not hit it).
        '''
        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        # NOTE: You will get this error:
        #   "TypeError: Swig director type mismatch in output value of
        #    type 'float32'"
        # without returning a value
        return fraction


class TDGroundArea(object):
    """
    An area on the ground that the car can run over
    """

    def __init__(self, friction_modifier):
        self.friction_modifier = friction_modifier


class TDTire(object):

    def __init__(self, car, max_forward_speed=25.0,
                 max_backward_speed=-20, max_drive_force=150,
                 turn_torque=40, max_lateral_impulse=3,
                 dimensions=(0.5, 1.25), density=1.0,
                 position=(0, 0)):

        world = car.body.world

        self.current_traction = 1
        self.turn_torque = turn_torque
        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed
        self.max_drive_force = max_drive_force
        self.max_lateral_impulse = max_lateral_impulse
        self.ground_areas = []

        self.body = world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(box=dimensions, density=density)
        self.body.userData = {'obj': self}

    @property
    def forward_velocity(self):
        body = self.body
        current_normal = body.GetWorldVector((0, 1))
        return current_normal.dot(body.linearVelocity) * current_normal

    @property
    def lateral_velocity(self):
        body = self.body

        right_normal = body.GetWorldVector((1, 0))
        return right_normal.dot(body.linearVelocity) * right_normal

    def update_friction(self):
        impulse = -self.lateral_velocity * self.body.mass
        if impulse.length > self.max_lateral_impulse:
            impulse *= self.max_lateral_impulse / impulse.length

        self.body.ApplyLinearImpulse(self.current_traction * impulse,
                                     self.body.worldCenter, True)

        aimp = 0.1 * self.current_traction * \
            self.body.inertia * -self.body.angularVelocity
        self.body.ApplyAngularImpulse(aimp, True)

        current_forward_normal = self.forward_velocity
        current_forward_speed = current_forward_normal.Normalize()

        drag_force_magnitude = -2 * current_forward_speed
        self.body.ApplyForce(self.current_traction * drag_force_magnitude * current_forward_normal,
                             self.body.worldCenter, True)

    def update_drive(self, keys):
        # if 'up' in keys:
        desired_speed = self.max_forward_speed
        if 'down' in keys:
            desired_speed = self.max_backward_speed
        # else:
        #     return

        # find the current speed in the forward direction
        current_forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = self.forward_velocity.dot(current_forward_normal)

        # apply necessary force
        force = 0.0
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        self.body.ApplyForce(self.current_traction * force * current_forward_normal,
                             self.body.worldCenter, True)

    def update_turn(self, keys):
        if 'left' in keys:
            desired_torque = self.turn_torque
        elif 'right' in keys:
            desired_torque = -self.turn_torque
        else:
            return

        self.body.ApplyTorque(desired_torque, True)

    def add_ground_area(self, ud):
        if ud not in self.ground_areas:
            self.ground_areas.append(ud)
            self.update_traction()

    def remove_ground_area(self, ud):
        if ud in self.ground_areas:
            self.ground_areas.remove(ud)
            self.update_traction()

    def update_traction(self):
        if not self.ground_areas:
            self.current_traction = 1
        else:
            self.current_traction = 0
            mods = [ga.friction_modifier for ga in self.ground_areas]

            max_mod = max(mods)
            if max_mod > self.current_traction:
                self.current_traction = max_mod


class TDCar(object):
    vertices = [(1.5, 0.0),
                (3.0, 2.5),
                (2.8, 5.5),
                (1.0, 10.0),
                (-1.0, 10.0),
                (-2.8, 5.5),
                (-3.0, 2.5),
                (-1.5, 0.0),
                ]

    tire_anchors = [(-3.0, 0.75),
                    (3.0, 0.75),
                    (-3.0, 8.50),
                    (3.0, 8.50),
                    ]

    def __init__(self, world, vertices=None,
                 tire_anchors=None, density=0.1, position=(0, 0),
                 **tire_kws):
        if vertices is None:
            vertices = TDCar.vertices

        self.body = world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(vertices=vertices, density=density)
        self.body.userData = {'obj': self}

        self.tires = [TDTire(self, **tire_kws) for i in range(4)]

        self.sensor_len = 50.0
        self.crashed = False
        self.hit_goal = False
        # self.fixing_crash = False
        # self.fixing_steps = 0
        self.calculate_sensors()
        self.sensor_sum = self.sensor_len * 5
        self.goal_sum = self.sensor_len * 5
        self.sensor_readings = [(self.sensor_len, None) for i in range(5)]
        self.goal_readings = [self.sensor_len for i in range(5)]
        self.norm_readings = [0 for i in range(5)]
        self.sensor_dist = [0.8, 1.0, 1.4, 1.0, 0.8]

        if tire_anchors is None:
            anchors = TDCar.tire_anchors

        joints = self.joints = []
        for tire, anchor in zip(self.tires, anchors):
            j = world.CreateRevoluteJoint(bodyA=self.body,
                                          bodyB=tire.body,
                                          localAnchorA=anchor,
                                          # center of tire
                                          localAnchorB=(0, 0),
                                          enableMotor=False,
                                          maxMotorTorque=1000,
                                          enableLimit=True,
                                          lowerAngle=0,
                                          upperAngle=0,
                                          )

            tire.body.position = self.body.worldCenter + anchor
            joints.append(j)

    def destroy(self, world):
        world.DestroyBody(self.body)
        for tire in self.tires:
            world.DestroyBody(tire.body)

    def calculate_sensors(self):
        d = (11 * math.cos(self.body.angle + math.radians(90)),
             11 * math.sin(self.body.angle + math.radians(90)))
        self.lidar = self.body.position + d

        self.sensor_tips = []
        self.sensor_tips.append(self.lidar + (self.sensor_len * math.cos(self.body.angle + math.radians(0)),
                                              self.sensor_len * math.sin(self.body.angle + math.radians(0))))
        self.sensor_tips.append(self.lidar + (self.sensor_len * math.cos(self.body.angle + math.radians(45)),
                                              self.sensor_len * math.sin(self.body.angle + math.radians(45))))
        self.sensor_tips.append(self.lidar + (self.sensor_len * math.cos(self.body.angle + math.radians(90)),
                                              self.sensor_len * math.sin(self.body.angle + math.radians(90))))
        self.sensor_tips.append(self.lidar + (self.sensor_len * math.cos(self.body.angle + math.radians(135)),
                                              self.sensor_len * math.sin(self.body.angle + math.radians(135))))
        self.sensor_tips.append(self.lidar + (self.sensor_len * math.cos(self.body.angle + math.radians(180)),
                                              self.sensor_len * math.sin(self.body.angle + math.radians(180))))

    def get_reward(self, goal, crashed, hit_goal):
        if crashed:
            if hit_goal:
                reward = 500
            else:
                reward = -750
        else:
            # reward = -10 + int(self.sensor_sum / 5)
            # reward = -20 + self.sensor_sum * self.sensor_sum / 100
            # min_sensor = min([x for (x, _) in self.sensor_readings])
            # reward = (200.0 - (self.body.position - goal).length) * 2 - (50 - min_sensor) * 5
            # reward = min_sensor * 2 - 50

            # reward = self.sensor_sum / 2 - self.goal_sum + 150 # 11/16

            sensor_sum = (250 - self.sensor_sum) * (-1 / 2)
            goal_sum = (250 - self.goal_sum) * (5 / 3)
            bias = -50 #10 if goal_sum == 0 else 0
            reward = (sensor_sum + goal_sum - bias) / 3.0
        return reward

    def reset(self):
        # tire_angles = [(tire, tire.body.angle) for tire in self.tires]
        body_angle = self.body.angle
        self.body.position = (0, 0)
        # self.body.angle = body_angle
        # for tire, angle in tire_angles:
        #     tire.body.angle = angle
        for tire in self.tires:
            tire.body.angle = self.body.angle

    def update(self, keys, hz, world, goal, goal_radius):

        for tire in self.tires:
            tire.update_friction()

        for tire in self.tires:
            tire.update_drive(keys)

        # control steering
        lock_angle = math.radians(40.)
        # from lock to lock in 0.5 sec
        turn_speed_per_sec = math.radians(160.)
        turn_per_timestep = turn_speed_per_sec / hz
        desired_angle = 0.0

        if 'left' in keys:
            desired_angle = lock_angle
        elif 'right' in keys:
            desired_angle = -lock_angle

        front_left_joint, front_right_joint = self.joints[2:4]
        angle_now = front_left_joint.angle
        angle_to_turn = desired_angle - angle_now

        # TODO fix b2Clamp for non-b2Vec2 types
        if angle_to_turn < -turn_per_timestep:
            angle_to_turn = -turn_per_timestep
        elif angle_to_turn > turn_per_timestep:
            angle_to_turn = turn_per_timestep

        new_angle = angle_now + angle_to_turn
        # Rotate the tires by locking the limits:
        front_left_joint.SetLimits(new_angle, new_angle)
        front_right_joint.SetLimits(new_angle, new_angle)

        # if self.crashed and not self.fixing_crash:
        #     self.fixing_crash = True
        #     self.fixing_steps = 0
        # if self.fixing_crash:
        #     self.fixing_steps += 1
        #     if self.fixing_steps > 100 and self.sensor_sum > 70:# and not self.crashed:
        #         self.fixing_crash = False
        #     elif self.fixing_steps > 100:
        #         print "Fixing didn't work, trying more"
        #         self.fixing_steps = 0

        self.calculate_sensors()

        # callback = RayCastClosestCallback()
        # self.sensor_readings = []
        # for i in range(len(self.sensor_readings) - 3):
        #     self.sensor_readings[i] = self.sensor_readings[i + 3]
        self.sensor_readings[0:len(self.sensor_readings)-5] = self.sensor_readings[5:len(self.sensor_readings)]
        del self.sensor_readings[-5:]
        self.goal_readings = []
        hitcnt = 0
        for pt in self.sensor_tips:
            callback = RayCastClosestCallback()
            world.RayCast(callback, self.lidar, pt)
            # dist = 0
            if callback.hit:# and callback.fixture.body != goal:
                hitcnt += 1
                # print str(callback.fixture.body)
                if callback.fixture.body != goal:
                    dist = (callback.point - self.lidar).length
                    self.sensor_readings.append((dist, callback.point))
                    # self.sensor_sum += dist
                    self.goal_readings.append(self.sensor_len)
                    # self.goal_sum += self.sensor_len
                else:
                    dist = (callback.point - self.lidar).length
                    self.sensor_readings.append((self.sensor_len, pt))
                    # self.sensor_sum += self.sensor_len
                    self.goal_readings.append(dist)
                    # self.goal_sum += dist
            # elif callback.hit and callback.fixture.body == goal:
            #     dist = (callback.point - self.lidar).length
            #     self.sensor_readings.append((self.sensor_len, pt))
            #     self.goal_readings.append(dist)
            #     self.goal_sum -= self.sensor_len - dist
            else:
                self.sensor_readings.append((self.sensor_len, pt))
                # self.sensor_sum += self.sensor_len
                self.goal_readings.append(self.sensor_len)
                # self.goal_sum += self.sensor_len

        if (self.lidar - goal.position).length < goal_radius:
            self.goal_readings = [0 for i in range(5)]
            self.goal_sum = 0

        self.norm_readings = []
        for (wall, _), goal in zip(self.sensor_readings, self.goal_readings):
            if goal != self.sensor_len:
                val = self.sensor_len - goal
            else:
                val = (self.sensor_len - wall) * -1
            self.norm_readings.append(val / self.sensor_len)

        self.sensor_sum = sum([dist * weight for (dist, _), weight in zip(self.sensor_readings, self.sensor_dist)])
        self.goal_sum = sum([dist * weight for dist, weight in zip(self.goal_readings, self.sensor_dist)])
        # print hitcnt
        # print "goal " + str(self.goal_readings)
        # print "sensor " + str([dist for dist, _ in self.sensor_readings])
        # time.sleep(0.2)



class TopDownCar(Framework):
    name = "Top Down Car"
    description = "Keys: accel = w, reverse = s, left = a, right = d"

    def __init__(self):
        super(TopDownCar, self).__init__()
        # Top-down -- no gravity in the screen plane
        self.world.gravity = (0, 0)

        self.key_map = {Keys.K_w: 'up',
                        Keys.K_s: 'down',
                        Keys.K_a: 'left',
                        Keys.K_d: 'right',
                        }

        # Keep track of the pressed keys
        self.pressed_keys = set()

        # The walls
        worldSize = 100
        boundary = self.world.CreateStaticBody(position=(0, 20))
        boundary.CreateEdgeChain([(-worldSize, -worldSize),
                                  (-worldSize, worldSize),
                                  (worldSize, worldSize),
                                  (worldSize, -worldSize),
                                  (-worldSize, -worldSize)]
                                 )

        self.goal_radius = 14

        # A couple regions of differing traction
        self.car = TDCar(self.world, position=(-5, 0))
        gnd1 = self.world.CreateStaticBody(userData={'obj': TDGroundArea(0.5)})
        fixture = gnd1.CreateCircleFixture(
            radius=16, friction=0.2, density=1.0)
            # box=(16, 18, (-40, 60), math.radians(-5)))#20)))
        gnd1.position = (-40, 60)

        gnd2 = self.world.CreateStaticBody(userData={'obj': TDGroundArea(0.2)})
        fixture = gnd2.CreateCircleFixture(
            radius=14, friction=0.2, density=1.0)
            # box=(12, 14, (-40, -20), math.radians(10)))#-40)))
        gnd2.position = (40, -30)

        gnd3 = self.world.CreateStaticBody(userData={'obj': TDGroundArea(0.2)})
        fixture = gnd3.CreateCircleFixture(
            radius=16, friction=0.2, density=1.0)
            # box=(12, 14, (40, -30), math.radians(30)))#-40)))
        gnd3.position = (-40, -10)

        gnd4 = self.world.CreateStaticBody(userData={'obj': TDGroundArea(0.2)})
        fixture = gnd4.CreateCircleFixture(
            radius=14, friction=0.2, density=1.0)
            # box=(12, 14, (50, 40), math.radians(45)))#-40)))
        gnd4.position = (50, 40)

        self.goal = self.world.CreateDynamicBody()
        fixture = self.goal.CreateCircleFixture(
            radius=self.goal_radius, friction=0.0, density=1.0)
            # box=(12, 14, (40, -20), math.radians(0)))#-40)))
        self.goal.position = (20, 90)
        fixture.sensor = True

        self.learner = Learner(self.car)
        self.obstacles = [gnd1, gnd2, gnd3, gnd4]
        self.steps = 0

    def Keyboard(self, key):
        key_map = self.key_map
        # if key in key_map:
        #     self.pressed_keys.add(key_map[key])
        # else:
        super(TopDownCar, self).Keyboard(key)

    def KeyboardUp(self, key):
        key_map = self.key_map
        if key is Keys.K_s:
            self.learner.save_brain()
        elif key is Keys.K_l:
            self.learner.load_brain()

        # if key in key_map:
        #     self.pressed_keys.remove(key_map[key])
        # else:
        super(TopDownCar, self).KeyboardUp(key)

    def handle_contact(self, contact, began):
        # A contact happened -- see if a wheel hit a
        # ground area
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        body_a, body_b = fixture_a.body, fixture_b.body
        ud_a, ud_b = body_a.userData, body_b.userData
        if not ud_a or not ud_b:
            return

        tire = None
        ground_area = None
        for ud in (ud_a, ud_b):
            obj = ud['obj']
            if isinstance(obj, TDTire):
                tire = obj
            elif isinstance(obj, TDGroundArea):
                ground_area = obj

        if ground_area is not None and tire is not None:
            if began:
                tire.add_ground_area(ground_area)
            else:
                tire.remove_ground_area(ground_area)

    def BeginContact(self, contact):
        # print "Watch where you're fuckin' going"
        self.handle_contact(contact, True)
        self.car.crashed = True
        if contact.fixtureA.body == self.goal or contact.fixtureB.body == self.goal:
            if (self.car.lidar - self.goal.position).length < self.goal_radius:
                self.car.hit_goal = True
            else:
                self.car.crashed = False

    def EndContact(self, contact):
        self.handle_contact(contact, False)
        self.car.crashed = False
        self.car.hit_goal = False

    def Step(self, settings):

        # PREDICT
        # qval = None
        # fixing = False
        translating = False
        if self.car.crashed:
            for obstacle in self.obstacles:
                obstacle.fixtures[0].sensor = True
            self.car.body.transform = (self.car.body.position * -1, 0)
            self.car.body.angle = math.radians(np.random.randint(0, 359))
            translating = True
        else:
            for obstacle in self.obstacles:
                obstacle.fixtures[0].sensor = False
            self.steps += 1
            self.pressed_keys = set()
            action = self.learner.take_action(self.steps)
            if action == 0:
                pass
            elif action == 1:
                self.pressed_keys.add('left')
            elif action == 2:
                self.pressed_keys.add('right')

        # UPDATE
        was_crashed = self.car.crashed
        was_hitting_goal = self.car.hit_goal
        self.car.update(self.pressed_keys, settings.hz, self.world, self.goal, self.goal_radius)
        super(TopDownCar, self).Step(settings)
        just_crashed = not was_crashed and self.car.crashed
        just_hit_goal = not was_hitting_goal and self.car.hit_goal

        if translating:
            body_angle = self.car.body.angle
            for tire in self.car.tires:
                tire.body.angle = body_angle

        self.learner.reinforce(self.steps, just_crashed, just_hit_goal, self.goal.position)

        # easy just count number of goals hit
        # if self.steps % 15000 == 0:

        # DRAW
        self.renderer.DrawPoint(self.renderer.to_screen(self.car.lidar), 5.0, b2Color(0.4, 0.9, 0.4))
        for pt in self.car.sensor_tips:
            self.renderer.DrawSegment(self.renderer.to_screen(self.car.lidar), self.renderer.to_screen(pt), b2Color(0.4, 0.9, 0.4))

        for dist, pt in self.car.sensor_readings:
            if dist != self.car.sensor_len:
                self.renderer.DrawPoint(self.renderer.to_screen(pt), 5.0, b2Color(0.4, 0.9, 0.4))


class Learner():

    def __init__(self, car):
        self.epsilon = 1.0
        self.gamma = 0.9
        self.buffer = 80
        self.batchSize = 40
        self.step_size = 8
        self.goals_needed = 5
        self.input_size = 5
        self.temp = 70

        self.goals_hit = 0
        self.car = car
        self.model = neural_net.neural_net(self.input_size)
        self.crashed = False
        self.replay = []
        self.h = 0
        self.lastQval = []
        self.lastEval = 0
        self.just_crashed = False
        self.just_hit_goal = False
        self.readings = None
        self.action = 0
        self.reinforced_for = []
        self.random = False
        self.all_rewards = []
        # self.file = open('rewards.dat', 'w')
        self.started = datetime.datetime.now()

    def take_action(self, step):
        if step - self.lastEval >= self.step_size:
            self.lastEval = step
            self.just_crashed = False
            self.just_hit_goal = False
            # self.readings = [dist for dist, _ in self.car.sensor_readings] + self.car.goal_readings
            self.readings = self.car.norm_readings
            self.lastQval = self.model.predict(np.array(self.readings).reshape(1, self.input_size), batch_size=1)
            if random.random() < self.epsilon:
                # self.action = np.random.randint(0, 3)
                # print str(np.arange(0, 3)) + " " + str(self.softmax(self.lastQval).tolist()[0])
                softmax = self.softmax(self.lastQval)
                # print str(self.lastQval)
                # print str(softmax)
                # print repr(np.sum(softmax))
                # self.action = np.random.choice(np.arange(0, 3), p=softmax.tolist()[0])
                self.action = self.choose_softmax(range(3), softmax.tolist()[0])
                self.random = True
            else:
                self.action = (np.argmax(self.lastQval))
                self.random = False
        return self.action

    def reinforce(self, step, just_crashed, just_hit_goal, goal_position):
        if step - self.lastEval < self.step_size - 1:
            if just_crashed:
                self.just_crashed = just_crashed
            if just_hit_goal:
                self.just_hit_goal = just_hit_goal
        elif step - self.lastEval == self.step_size - 1 and step not in self.reinforced_for:
            # print "reinforcing at " + str(step)
            self.reinforced_for.append(step)
            if len(self.reinforced_for) > 10:
                del self.reinforced_for[:5]
            reward = self.car.get_reward(goal_position, self.just_crashed or just_crashed, self.just_hit_goal or just_hit_goal)
            if self.just_hit_goal or just_hit_goal:
                self.goals_hit += 1
                self.update_epsilon()
            # self.all_rewards.append(reward)
            # self.file.write(str(reward) + "\n")
            # to_debug += " " + str(reward) + " (" + str(self.epsilon) + ")"
            # print to_debug
            if self.lastEval != 0:
                newQ = self.model.predict(np.array(self.car.norm_readings).reshape(1, self.input_size), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1, 3))
                y[:] = self.lastQval[:]
                if reward == 500 or reward == -750:
                    update = reward
                else:
                    update = (reward + (self.gamma * maxQ))
                y[0][self.action] = update
                self.model.fit(np.array(self.readings).reshape(1,self.input_size), y, batch_size=1, nb_epoch=1, verbose=0)
                self.print_debug(reward)

            # if len(self.replay) < self.buffer:
            #     if self.readings is not None:
            #         # self.replay.append((self.readings, self.action, reward, [dist for dist, _ in self.car.sensor_readings] + self.car.goal_readings))
            #         self.replay.append((self.readings, self.action, reward, self.car.norm_readings))
            #         self.print_debug(reward)
            # else:
            #     if self.h < (self.buffer - 1):
            #         self.h += 1
            #     else:
            #         self.h = 0
            #     # self.replay[self.h] = (self.readings, self.action, reward, [dist for dist, _ in self.car.sensor_readings] + self.car.goal_readings)
            #     self.replay[self.h] = (self.readings, self.action, reward, self.car.norm_readings)
            #     self.print_debug(reward)
            #     minibatch = random.sample(self.replay, self.batchSize)
            #     X_train = []
            #     y_train = []
            #     # print "{"
            #     for old_state, action, old_reward, new_state in minibatch:
            #         old_qval = self.model.predict(np.array(old_state).reshape(1, self.input_size), batch_size=1)
            #         newQ = self.model.predict(np.array(new_state).reshape(1, self.input_size), batch_size=1)
            #         maxQ = np.max(newQ)
            #         y = np.zeros((1, 3))
            #         y[:] = old_qval[:]
            #         if old_reward == 1000 or old_reward == -15000:
            #             update = old_reward
            #         else:
            #             update = (old_reward + (self.gamma * maxQ))
            #         y[0][action] = update
            #         X_train.append(np.array(old_state).reshape(self.input_size,))
            #         y_train.append(y.reshape(3,))
            #     #     print str(old_state)
            #     #     print str(new_state)
            #     #     print str(reward)
            #     # print "}"
            #     # time.sleep(0.3)
            #
            #     X_train = np.array(X_train)
            #     y_train = np.array(y_train)
            #     self.model.fit(X_train, y_train, batch_size=self.batchSize, nb_epoch=1, verbose=0)

    def update_epsilon(self):
        if self.goals_hit % self.goals_needed == 0:
            if self.epsilon > 0.1:
                self.goals_needed += 1
                self.goals_hit = 0
                self.epsilon -= 0.05
            else:
                self.save_brain()
                end = datetime.datetime.now()
                print str(end)
                print str(end - self.started)
                # f = open('rewards.dat', 'w')
                # f.write(str(self.all_rewards))
                # self.file.close()
                sys.exit(0)

    def save_brain(self):
        self.model.save_weights('saved-models/brain.h5', overwrite=True)
        print "Saved model"

    def load_brain(self):
        self.model.load_weights('saved-models/brain.h5')
        print "Loaded model"

    def print_debug(self, reward):
        to_print = "[*]" if self.random else "[!]"
        to_print += " action " + str(self.action) + " reward " + str(reward) + " epsilon " + str(self.epsilon) +\
                    " goals " + str(self.goals_hit) + "/" + str(self.goals_needed)
        # to_print = str(self.car.norm_readings)
        to_print += " softmax " + str(self.softmax(self.lastQval))
        to_print += " inputs " + str(self.car.norm_readings)
        print to_print

    def softmax(self, l):
        # l = k + k.min()
        softmax = np.exp(l / self.temp) / np.sum(np.exp(l / self.temp))
        softmax /= softmax.sum().astype(float)
        # print repr(softmax.sum())
        # dif = float(repr(softmax.sum())) - 1.0
        # softmax[0] -= dif
        # print repr(dif)
        return softmax / np.sum(softmax)

    def choose_softmax(self, choices, weights):
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for c, w in zip(choices, weights):
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"


if __name__ == "__main__":
    __author__ = 'ianperry'
    main(TopDownCar)
