# COMP532 Assignment 1
# Phoebe Edwards (200786023)
# Matt Fishwick (201531430)

# Problem 1

import numpy as np
import matplotlib.pyplot as plt 

noActions = 10
epsilon = [0, 0.1, 0.01]
noSteps = 1000
noIterations = 2000
actionValues = np.zeros(noActions)

rewardsArray = np.zeros((noSteps, len(epsilon)))
optimalActionArray = np.zeros((noSteps, len(epsilon)))

policyCounter = -1
# Loop for each value of epsilon
for policy in epsilon:
    policyCounter += 1
    print("Running: epsilon =", policy)

    for iteration in range(noIterations):
        
        actionValues = np.random.normal(loc=0, scale=1, size=noActions) # Set random normal values with mean = 0 and standard deviation = 1
        optimalAction = np.argmax(actionValues)

        rewardsSum = np.zeros(noActions)
        actionCount = np.zeros(noActions)
        valueEstimates = np.zeros(noActions) 

        # Loop for number of steps in an iteration
        for step in range(1, noSteps):
    
            # take action based on epsilon greedy policy
            randomProbability = np.random.random()
            if(randomProbability < policy):
                # select random action
                actionAt_t = np.random.randint(noActions)
            else:
                greedyAction = np.argmax(valueEstimates)
                possibleActions = np.where(valueEstimates == valueEstimates[greedyAction])[0]

                # if multiple actions have the same value, randomly select an action
                if(len(possibleActions) <= 1):
                    actionAt_t = greedyAction
                else:
                    actionAt_t = np.random.choice(possibleActions)

            # initialise rewards
            rewards = np.random.normal(loc=actionValues, scale=1, size=noActions)

            rewardRecieved = rewards[actionAt_t]
            rewardsSum[actionAt_t] += rewardRecieved  
            actionCount[actionAt_t] += 1               
            valueEstimates[actionAt_t] = rewardsSum[actionAt_t]/actionCount[actionAt_t]

            # data for graph 1
            rewardsArray[step, policyCounter] += rewardRecieved

            # data for graph 2
            if(actionAt_t == optimalAction):
                optimalActionArray[step, policyCounter] += 1

rewardAverage = rewardsArray/noIterations
optimalAverage = optimalActionArray/noIterations

#Graph 1 - average rewards over all steps
plt.title("Average Rewards")
plt.plot(rewardAverage)
plt.ylim(0, 1.5)
plt.ylabel("Average Reward")
plt.xlabel("Steps")
plt.show()

#Graph 2 - optimal selections over all steps
plt.title("% Optimal Action")
plt.plot(optimalAverage * 100)
plt.ylim(0, 100)
plt.ylabel("% Optimal Action")
plt.xlabel('Steps')
plt.show()