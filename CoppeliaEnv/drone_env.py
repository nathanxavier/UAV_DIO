# -*- coding: utf-8 -*-
import numpy as np
import sim
import gym
from gym import spaces

class DroneEnv(gym.Env):
    metadata = {"render_modes": [None, "human"]} #Environment is continously
    def __init__(self, modeCoppelia, ctrlAng=True, ctrlPos=True, render_mode="human", size=1, epsilon=1e-2, maxTime=10):
        self.ctrlPos = ctrlPos      # Variação de Posição
        self.ctrlAng = ctrlAng      # Ângulo de Ângulo
        self.mode = modeCoppelia    # Modo de atuação do Coppelia
        self.size = size            # Tamanho do espaço de decolagem
        self.f_stable = 0           # Flag de estabilidade do Drone
        self.tempSim = 0            # Tempo de Simulação
        self.epsilon = epsilon      # Erro mínimo aceitável

        self.minPosDrone = 1e-2
        self.maxPosDrone = 10
        
        self.minPosTarg = 0
        self.maxPosTarg = 10
        ''' Espaço de Observação
        Limites do Agente e Target -> x, y, z, Roll, Pitch, Yaw
            Localização X-Y = [-size, size]
            Localização Z:
                Agent = [1e-2, 10]
                Target = [1, 10]
            Ângulo Roll-Pitch-Yaw = [-pi, pi]
        
        self.observation_space = spaces.Dict(
            {
                "agentPos": spaces.Box(low=np.array([-self.size, -self.size, self.minPosDrone]),
                                       high=np.array([self.size, self.size, self.maxPosDrone]),
                                       dtype=float),
                "agentAng": spaces.Box(low=np.array([-np.pi, -np.pi, -np.pi]),
                                        high=np.array([-np.pi, -np.pi, np.pi]),
                                        dtype=float),
                "targetPos": spaces.Box(low=np.array([-self.size, -self.size, self.minPosTarg]),
                                        high=np.array([self.size, self.size, self.maxPosTarg]),
                                        dtype=float),
                "targetAng": spaces.Box(low=np.array([-np.pi, -np.pi, -np.pi]),
                                        high=np.array([-np.pi, -np.pi, np.pi]),
                                        dtype=float),
            }
        )
        '''
        # Considerando o Target sempre Fixo e o Drone em um cubo de 1un³
        self.posTarg = 0
        self.observation_space = spaces.Dict(
            {
                "agentPos": spaces.Box(low=np.array([-self.size, -self.size, self.posTarg-self.size]),
                                       high=np.array([self.size, self.size, self.posTarg+self.size]),
                                       dtype=float),
                "agentAng": spaces.Box(low=np.array([-np.pi, -np.pi, -np.pi]),
                                        high=np.array([-np.pi, -np.pi, np.pi]),
                                        dtype=float),
                "targetPos": spaces.Box(low=np.array([0,0,self.posTarg]),
                                        high=np.array([0,0,self.posTarg]),
                                        dtype=float),
                "targetAng": spaces.Box(low=np.array([-np.pi, -np.pi, -np.pi]),
                                        high=np.array([-np.pi, -np.pi, np.pi]),
                                        dtype=float),
            }
        )

        '''Espaço de Ações
            Ações Contínuas
                4 PWM = [0, 100]
        '''
        self.minPWM = 0
        self.maxPWM = 100
        self.action_space = spaces.Box(low=np.array([self.minPWM,self.minPWM,self.minPWM,self.minPWM]),
                                       high=np.array([self.maxPWM,self.maxPWM,self.maxPWM,self.maxPWM]),
                                       dtype=float)
        
        self.render_mode = render_mode
        
        # Parâmetros para o cálculo do Reward
        self.pos0 = 0
        self.ang0 = 0
        self.Dist0 = 0
        self.lastPos = 0
        self.lastOri = 0

        self.clock = None
        self.timeStep = None
        self.maxTime = maxTime
        # Coppelia
        sim.simxFinish(-1) # just in case, close all opened connections
        
        self.clientID = sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
        sim.simxSynchronous(self.clientID, True) # Modo de Steps
        self.robotname = 'Quadcopter'
        
        # Paralização dos motores
        ctrlSign = sim.simxPackFloats(np.array([0,0,0,0]))
        sim.simxSetStringSignal(self.clientID, "controlSignal", ctrlSign, sim.simx_opmode_oneshot_wait)

        # Ativação da ação dos motores
        sim.simxSetInt32Signal(self.clientID, "Start", self.mode, sim.simx_opmode_oneshot_wait)

        returnCode, self.robotHandle = sim.simxGetObjectHandle(self.clientID, self.robotname, sim.simx_opmode_oneshot_wait)
        returnCode, self.target = sim.simxGetObjectHandle(self.clientID, '/target', sim.simx_opmode_oneshot_wait)

    def readSensors(self):
        # Leituras dos sensores
        returnCode, self._agent_location = sim.simxGetObjectPosition(self.clientID, self.robotHandle, -1, sim.simx_opmode_oneshot_wait)
        returnCode, self._agent_orientation = sim.simxGetObjectOrientation(self.clientID, self.robotHandle, -1, sim.simx_opmode_oneshot_wait)
        returnCode, self._target_location = sim.simxGetObjectPosition(self.clientID, self.target, -1, sim.simx_opmode_oneshot_wait)
        returnCode, self._target_orientation = sim.simxGetObjectOrientation(self.clientID, self.target, -1, sim.simx_opmode_oneshot_wait)

        self._agent_location = np.array(self._agent_location)
        self._agent_orientation = np.array(self._agent_orientation)
        self._target_location = np.array(self._target_location)
        self._target_orientation = np.array(self._target_orientation)

        return np.ravel([self._agent_location,
                         self._agent_orientation, 
                         self._target_location,
                         self._target_orientation])
    
    def readActuators(self):
        returnCode, ctrlSign = sim.simxGetStringSignal(self.clientID, "controlSignal", sim.simx_opmode_oneshot_wait)
        action = sim.simxUnpackFloats(ctrlSign)

        return np.array(action)

    def calcReward(self):
        self.distOrig = np.linalg.norm(self._agent_location -self.pos0)
        self.distAnt = np.linalg.norm(self._target_location -self.lastPos)
        self.distAtual = np.linalg.norm(self._target_location -self._agent_location)
        self.deltaPos = np.linalg.norm(self._agent_location -self.lastPos)
        self.angAtual = np.arctan2(self._agent_location, self._target_location) -self._agent_orientation
        self.angAnt = np.arctan2(self.lastPos, self._target_location) -self.lastOri

        '''Recompensas:
            Área [-inf, 1]
            Posição [-inf, 1]
            dPos [-inf, +inf]
            Vel [-inf, 0]
            Ang [-inf, 0]
            Omega [-inf, 0]
        '''
        rewardArea = ((self.Dist0 +self.epsilon) -(self.distOrig +self.distAtual))
        rewardPos = (self.Dist0 -self.distAtual +self.epsilon) / self.Dist0
        rewardDeltaPos = self.distAnt -self.distAtual
        rewardVelX = -abs(self._agent_location[0] -self.lastPos[0]) /(int(np.rint(.05/self.timeStep)) *self.timeStep)
        rewardVelY = -abs(self._agent_location[1] -self.lastPos[1]) /(int(np.rint(.05/self.timeStep)) *self.timeStep)
        rewardVelZ = -abs(self._agent_location[2] -self.lastPos[2]) /(int(np.rint(.05/self.timeStep)) *self.timeStep)
        
        rewardVel = -self.deltaPos /.05 # self.timeStep

        rewardAng = -np.sum(abs(self._agent_orientation -self._target_orientation)) / (2*np.pi)
        rewardRoll = -abs(self._agent_orientation[0] -self.lastOri[0])
        rewardPitch = -abs(self._agent_orientation[1] -self.lastOri[1])
        rewardYaw = -abs(self._agent_orientation[2] -self.lastOri[2])
        rewardOmegaRoll = -abs(self._agent_orientation[0] -self.lastOri[0]) /(int(np.rint(.05/self.timeStep)) *self.timeStep)
        rewardOmegaPitch = -abs(self._agent_orientation[1] -self.lastOri[1]) /(int(np.rint(.05/self.timeStep)) *self.timeStep)
        rewardOmegaYaw = -abs(self._agent_orientation[2] -self.lastOri[2]) /(int(np.rint(.05/self.timeStep)) *self.timeStep)
        # weights_Ang = np.array([.01,.01,.5]).reshape([1,3])
        # rewardOri = np.dot(weights_Ang, self.angAtual-self.angAnt)[0]
        # if(abs(self.lastDist -self.distAtual)>.01):
        #     rewardDeltaPos = -abs(rewardDeltaPos)

        '''Peso das Recompensas'''
        # reward_weights = np.array([1, 1, 1, .1, .1, .1, .1, .1])
        # reward_weights = reward_weights/np.sum(reward_weights) *len(reward_weights)
        # self.rewards = np.array([rewardArea, rewardPos, rewardDeltaPos, rewardVel, rewardAng, rewardRoll, rewardPitch, rewardYaw,])

        reward_weights = np.array([1,
                                   .1, .1, .1, 
                                   .5, .5, .5])
        # reward_weights = reward_weights/np.sum(reward_weights) *len(reward_weights)
        self.rewards = np.array([rewardPos, 
                                 rewardVelX, rewardVelY, rewardVelZ, 
                                 rewardOmegaRoll, rewardOmegaPitch, rewardOmegaYaw,])

        rewardTotal = np.dot(reward_weights, self.rewards)
        rewardTotal = np.dot(reward_weights, np.max([self.rewards,-10*np.ones(len(self.rewards))], axis=0))
        
        # print("Reward:", np.round(reward_weights[0]*self.rewards[0], 2),
        #                  np.round(reward_weights[1]*self.rewards[1], 2),
        #                  np.round(reward_weights[2]*self.rewards[2], 2),
        #                  np.round(reward_weights[3]*self.rewards[3], 2),
        #                  np.round(reward_weights[4]*self.rewards[4], 2),
        #                 #  np.round(self.rewards, 2),
        #                  rewardTotal
        #     #   np.round(rewardArea, 2),
        #     #   np.round(rewardPos, 2),
        #     #   np.round(rewardDeltaPos, 2),
        #     #   np.round(rewardVel, 2),
        #     #   np.round(rewardAng, 2),
        #     #   np.round(rewardRoll, 2),
        #     #   np.round(rewardPitch, 2),
        #     #   np.round(rewardYaw, 2),
        #     #   np.round(rewardOri, 2),
        #     #   np.round(self.angAtual-self.angAnt, 2),
        #       )

        return rewardTotal
        
        # return np.max([np.min([reward_weights[0]*rewardArea + 
        #                        reward_weights[1]*rewardPos + 
        #                        reward_weights[2]*rewardDeltaPos + 
        #                        reward_weights[3]*rewardVel +
        #                        reward_weights[4]*rewardAng + 
        #                        reward_weights[5]*rewardOmega, len(reward_weights)]), -len(reward_weights)])

    def _get_obs(self):
        return {"agentPos": self._agent_location,
                "agentAng": self._agent_orientation,
                "targetPos": self._target_location,
                "targetAng": self._target_orientation,
                }
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location -self._target_location),
                "angle": self._agent_orientation[-1] -self._target_orientation[-1]}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Bloqueio da ação dos motores
        sim.simxSetInt32Signal(self.clientID, "Start", 0, sim.simx_opmode_oneshot_wait)
        
        if(self.render_mode == "human"):
            self._render_frame()
        
        # Localização e Orientação aleatória do drone no plano X-Y
        if(self.ctrlAng):
            self._agent_orientation = self.np_random.uniform(low=self.observation_space['agentAng'].low,
                                                             high=self.observation_space['agentAng'].high)
            self._target_orientation = self.np_random.uniform(low=self.observation_space['targetAng'].low,
                                                              high=self.observation_space['targetAng'].high)
        else:
            self._agent_orientation = self.np_random.uniform(low=np.array([0,0,0]),
                                                             high=np.array([0,0,0]))
            self._target_orientation = self.np_random.uniform(low=np.array([0,0,0]),
                                                              high=np.array([0,0,0]))
        
        if(self.ctrlPos):
             self._agent_location = self.np_random.uniform(low=np.array([-1,-1,-1]),
                                                           high=np.array([1,1,1]))
             self._target_location = self.np_random.uniform(low=self.observation_space['targetPos'].low,
                                                            high=self.observation_space['targetPos'].high)
        else:
             self._agent_location = self.np_random.uniform(low=np.array([0,0,self.posTarg-1]),
                                                           high=np.array([0,0,self.posTarg-1]))
             self._target_location = self.np_random.uniform(low=np.array([0,0,self.posTarg]),
                                                            high=np.array([0,0,self.posTarg]))
        
        '''Reset no Coppelia'''        
        # Reset da Orientação
        returnCode = sim.simxSetObjectOrientation(self.clientID, self.robotHandle, -1, self._agent_orientation, sim.simx_opmode_oneshot_wait)
        returnCode = sim.simxSetObjectOrientation(self.clientID, self.target, -1, self._target_orientation, sim.simx_opmode_oneshot_wait)

        # Reset de Posição
        returnCode = sim.simxSetObjectPosition(self.clientID, self.robotHandle, -1, self._agent_location, sim.simx_opmode_oneshot_wait)
        returnCode = sim.simxSetObjectPosition(self.clientID, self.target, -1, self._target_location, sim.simx_opmode_oneshot_wait)
        
        # Paralização dos Motores
        ctrlSign = sim.simxPackFloats(np.array([0,0,0,0]))
        sim.simxSetStringSignal(self.clientID, "controlSignal", ctrlSign, sim.simx_opmode_oneshot_wait)

        # Atualização das distâncias e ângulos
        self.pos0 = self._agent_location.copy()
        self.lastPos = self._agent_location.copy()
        self.lastOri = self._agent_orientation.copy()
        self.Dist0 = np.linalg.norm(self._target_location -self.pos0)
        self.tempSim = 0

        observation = self._get_obs()
        info = self._get_info()
        
        # # Desbloqueio motores
        # sim.simxSetInt32Signal(self.clientID, "Start", self.mode, sim.simx_opmode_oneshot_wait)

        return observation, info
    
    def step(self, action):
        if(self.mode==1):
            action = self.readActuators()
        elif(self.mode==2):
            # Ação do Motor
            ctrlSign = sim.simxPackFloats(action)
            sim.simxSetStringSignal(self.clientID, "controlSignal", ctrlSign, sim.simx_opmode_oneshot_wait)

        # Leituras do Coppelia
        states = self.readSensors()

        # Recompensa
        reward = self.calcReward()
        
        observation = self._get_obs()
        info = self._get_info()
        
        self.tempSim += 1

        if(self.render_mode == "human"):
            self._render_frame()
        
        # Condição de Parada Precoce
        if(self.tempSim>self.maxTime/self.timeStep):
            # Demorou muito para alcançar o Target
            truncated = True
            reward -= 10
        elif(self.distAtual >2*(self.size)):
            # or (abs(self._agent_orientation[0:2])>np.pi/2).any()):
            # Drone caiu da base ou Drone invertido
            truncated = True
            # reward -= self.maxTime/self.timeStep
        # elif((self.distAtual >1.1*self.size).any()):
            # Drone saiu da base, mas continua voando
            # truncated = False
            # reward -= 1
        else:
            truncated = False
            reward += 10 # Recompensa por estar vivo
        
        # Target alcançado
        if((self.distAtual<self.epsilon) and
           (abs(self._agent_orientation[-1] -self._target_orientation[-1])<self.epsilon)):
            self.f_stable +=1
            reward += self.timeStep
            terminated = False
            if(self.f_stable >= 1/self.timeStep):
                # reward += 10
                terminated = True
        else:
            self.f_stable = 0
            terminated = False
        
        self.lastPos = self._agent_location.copy()
        self.lastOri = self._agent_orientation.copy()

        return observation, reward, terminated, truncated, info
            
    def _render_frame(self):
        '''Dados de Frame do Simulador'''
        returnCode, self.clock = sim.simxGetFloatSignal(self.clientID, "simulationTime", sim.simx_opmode_oneshot_wait)
        returnCode, self.timeStep = sim.simxGetFloatSignal(self.clientID, "simulationTimeStep", sim.simx_opmode_oneshot_wait)

        '''Step no Simulador'''
        for _ in range(int(np.rint(.05/self.timeStep))):
            returnCode = sim.simxSynchronousTrigger(self.clientID)

    def set_mode(self, modeCoppelia, ctrlAng=True, ctrlPos=True):
        '''Modos de Operação do Coppelia
                0 = Bloqueio da ação dos motores
                1 = Automática (Padrão Simulador)
                2 = Manual
        '''
        self.ctrlPos = ctrlPos   # Variação de Posição
        self.ctrlAng = ctrlAng   # Ângulo de Ângulo
        self.mode = modeCoppelia # Modo de atuação do Coppelia
        sim.simxSetInt32Signal(self.clientID, "Start", self.mode, sim.simx_opmode_oneshot_wait)

        if(self.render_mode == "human"):
            self._render_frame()

        returnCode, self.clock = sim.simxGetFloatSignal(self.clientID, "simulationTime", sim.simx_opmode_oneshot_wait)
        returnCode, self.timeStep = sim.simxGetFloatSignal(self.clientID, "simulationTimeStep", sim.simx_opmode_oneshot_wait)
        
    def force_auto(self):
        sim.simxSetInt32Signal(self.clientID, "Start", 0, sim.simx_opmode_oneshot_wait)
        returnCode = sim.simxSetObjectOrientation(self.clientID, self.robotHandle, -1, [0, 0, 0], sim.simx_opmode_oneshot_wait)
        returnCode = sim.simxSetObjectPosition(self.clientID, self.robotHandle, -1, [0, 0, 0], sim.simx_opmode_oneshot_wait)
        
        # self._render_frame()
        self.mode = 1
        sim.simxSetInt32Signal(self.clientID, "Start", self.mode, sim.simx_opmode_oneshot_wait)
    
    def restart(self):
        '''Finalização do Simulador'''
        # Bloqueio da ação dos motores
        sim.simxSetInt32Signal(self.clientID, "Start", 0, sim.simx_opmode_oneshot_wait)

        # Paralização dos motores
        ctrlSign = sim.simxPackFloats(np.array([0,0,0,0]))
        sim.simxSetStringSignal(self.clientID, "controlSignal", ctrlSign, sim.simx_opmode_oneshot_wait)
        
        # Reset da Orientação
        returnCode = sim.simxSetObjectOrientation(self.clientID, self.robotHandle, -1, [0,0,0], sim.simx_opmode_oneshot_wait)
        returnCode = sim.simxSetObjectOrientation(self.clientID, self.target, -1, [0,0,0], sim.simx_opmode_oneshot_wait)

        # Reset de Posição
        returnCode = sim.simxSetObjectPosition(self.clientID, self.robotHandle, -1, [0,0,self.posTarg], sim.simx_opmode_oneshot_wait)
        returnCode = sim.simxSetObjectPosition(self.clientID, self.target, -1, [0,0,self.posTarg], sim.simx_opmode_oneshot_wait)

        # if(self.render_mode == "human"):
        #     self._render_frame()

        # # Desbloqueio motores
        # sim.simxSetInt32Signal(self.clientID, "Start", self.mode, sim.simx_opmode_oneshot_wait)

        self.clock = None
        self.timeStep = None

    def close(self):
        '''Finalização do Simulador'''
        # Bloqueio da ação dos motores
        sim.simxSetInt32Signal(self.clientID, "Start", 0, sim.simx_opmode_oneshot_wait)
        
        # Paralização dos motores
        ctrlSign = sim.simxPackFloats(np.array([0,0,0,0]))
        sim.simxSetStringSignal(self.clientID, "controlSignal", ctrlSign, sim.simx_opmode_oneshot_wait)
        
        # Reset da Orientação
        returnCode = sim.simxSetObjectOrientation(self.clientID, self.robotHandle, -1, [0,0,0], sim.simx_opmode_oneshot_wait)
        returnCode = sim.simxSetObjectOrientation(self.clientID, self.target, -1, [0,0,0], sim.simx_opmode_oneshot_wait)

        # Reset de Posição
        returnCode = sim.simxSetObjectPosition(self.clientID, self.robotHandle, -1, [0,0,self.posTarg], sim.simx_opmode_oneshot_wait)
        returnCode = sim.simxSetObjectPosition(self.clientID, self.target, -1, [0,0,self.posTarg], sim.simx_opmode_oneshot_wait)
        
        # Finalização do Coppelia
        sim.simxFinish(self.clientID)
