#!/usr/bin/env python3


import numpy as np
import pygame as pg


# Initializing the Environment class

class Environment:
    def __init__(self, waitTime):

        # Defining the parameters
        self.width: int = 800    # Width of window
        self.height: int = 800   # Height of window
        self.nRows: int = 10     # Total rows on board
        self.nColumns: int = 10  # Total columns on board

        self.initSnakeLen: int = 2      # Initial length of snake
        self.defReward: float = -0.03   # Reward for taking an action - living penalty
        self.negReward: float = -1.0    # Reward for dying
        self.posReward: float = 2.0     # Reward for collecting an apple
        self.waitTime = waitTime        # Slowdown after taking an action

        if self.initSnakeLen > self.nRows / 2:
            self.initSnakeLen = int(self.nRows / 2)

        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("Board Snake 🐍 ")
        self.snakePos = list()

        # Creating the array that contains mathematical representation of the game's board
        self.screenMap = np.zeros((self.nRows, self.nColumns))

        # Set the initial snake on screen
        for i in range(self.initSnakeLen):
            self.snakePos.append([int(self.nRows / 2) + i, int(self.nColumns / 2)])  # Set position of snake part
            self.screenMap[int(self.nRows / 2) + i, int(self.nColumns / 2)] = 0.5    # Denote board location is covered by snake part

        self.applePos = self.placeApple()    # Place apple on board
        self.drawScreen()
        self.collected: bool = False         # Indicate whether apple is collected or not
        self.lastMove = 0

    # Building a method that gets new, random position of an apple
    def placeApple(self):
        randomGenerator = np.random.default_rng()
        colPosX = randomGenerator.integers(0, self.nColumns)
        rowPosY = randomGenerator.integers(0, self.nRows)
        while self.screenMap[rowPosY, colPosX] == 0.5:
            colPosX = randomGenerator.integers(0, self.nColumns)
            rowPosY = randomGenerator.integers(0, self.nRows)
        self.screenMap[rowPosY, colPosX] = 1.0
        return (rowPosY, colPosX)

    # Making a function that draws everything for us to see
    def drawScreen(self):
        self.screen.fill((0, 0, 0))
        cellWidth = self.width / self.nColumns
        cellHeight = self.height / self.nRows

        for i in range(self.nRows):
            for j in range(self.nColumns):
                if self.screenMap[i, j] == 0.5:
                    pg.draw.rect(self.screen, (255, 255, 255), (j * cellWidth + 1, i * cellHeight + 1, cellWidth - 2, cellHeight - 2))
                elif self.screenMap[i, j] == 1.0:
                    pg.draw.rect(self.screen, (255, 0, 0), (j * cellWidth + 1, i * cellHeight + 1, cellWidth - 2, cellHeight - 2))
        pg.display.flip()

    # A method that updates the snake's position
    def moveSnake(self, next_pos, is_collected: bool):
        self.snakePos.insert(0, next_pos)
        if not is_collected:
            self.snakePos.pop(len(self.snakePos) - 1)

        self.screenMap = np.zeros((self.nRows, self.nColumns))
        for i in range(len(self.snakePos)):
            self.screenMap[self.snakePos[i][0]][self.snakePos[i][1]] = 0.5

        if is_collected:
            self.applePos = self.placeApple()
            self.collected = True

        self.screenMap[self.applePos[0]][self.applePos[1]] = 1.0

    # The main method that updates the environment
    def step(self, action):
        # action 0 -> Up, 1 -> Down, 2 -> Right, 3 -> Left
        # Resetting these parameters and setting the reward to the living penalty
        gameOver = False
        reward = self.defReward    # Initially set living reward for any action
        self.collected = False

        for event in pg.event.get():
            if event == pg.QUIT:
                return

        snakeX = self.snakePos[0][1]
        snakeY = self.snakePos[0][0]

        # Checking if an action is playable and if not then it is changed to the playable one
        if action == 1 and self.lastMove == 0:
            action = 0
        elif action == 0 and self.lastMove == 1:
            action = 1
        elif action == 3 and self.lastMove == 2:
            action = 2
        elif action == 2 and self.lastMove == 3:
            action = 3

        # Checking what happens when we take this action
        if action == 0:
            if snakeY > 0:
                if self.screenMap[snakeY - 1][snakeX] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY - 1][snakeX] == 1.0:
                    reward = self.posReward
                    self.moveSnake(next_pos=(snakeY - 1, snakeX), is_collected=True)
                elif self.screenMap[snakeY - 1][snakeX] == 0:
                    self.moveSnake(next_pos=(snakeY - 1, snakeX), is_collected=False)
            else:
                gameOver = True
                reward = self.negReward

        elif action == 1:
            if snakeY < self.nRows - 1:
                if self.screenMap[snakeY + 1][snakeX] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY + 1][snakeX] == 1.0:
                    reward = self.posReward
                    self.moveSnake(next_pos=(snakeY + 1, snakeX), is_collected=True)
                elif self.screenMap[snakeY + 1][snakeX] == 0:
                    self.moveSnake(next_pos=(snakeY + 1, snakeX), is_collected=False)
            else:
                gameOver = True
                reward = self.negReward

        elif action == 2:
            if snakeX < self.nColumns - 1:
                if self.screenMap[snakeY][snakeX + 1] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY][snakeX + 1] == 1.0:
                    reward = self.posReward
                    self.moveSnake(next_pos=(snakeY, snakeX + 1), is_collected=True)
                elif self.screenMap[snakeX + 1][snakeY] == 0:
                    self.moveSnake(next_pos=(snakeY, snakeX + 1), is_collected=False)
            else:
                gameOver = True
                reward = self.negReward

        elif action == 3:
            if snakeX > 0:
                if self.screenMap[snakeY][snakeX - 1] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY][snakeX - 1] == 1.0:
                    reward = self.posReward
                    self.moveSnake(next_pos=(snakeY, snakeX - 1), is_collected=True)
                elif self.screenMap[snakeY][snakeX - 1] == 0:
                    self.moveSnake(next_pos=(snakeY, snakeX - 1), is_collected=False)
            else:
                gameOver = True
                reward = self.negReward

        # Drawing the screen, updating last move and waiting the wait time specified
        self.drawScreen()
        self.lastMove = action
        pg.time.wait(self.waitTime)

        # Returning the new frame of the game, the reward obtained and whether the game has ended or not
        return self.screenMap, reward, gameOver

    # Making a function that resets the environment
    def reset(self):
        self.snakePos = list()
        self.screenMap = np.zeros((self.nRows, self.nColumns))
        for i in range(self.initSnakeLen):
            self.screenMap[int(self.nRows / 2) + i, int(self.nColumns / 2)] = 0.5
            self.snakePos.append((int(self.nRows / 2) + i, int(self.nColumns / 2)))

        self.screenMap[self.applePos[0], self.applePos[1]] = 1.0
        self.collected = False
        self.lastMove = 0

