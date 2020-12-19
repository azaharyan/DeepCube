from cubeAgent import CubeAgent

def adi(iterations=10):
    for _ in range(iterations):

        # generate N scrambled cubes
        cubes = CubeAgent(number_of_cubes=100)
        cubes.scrabmle_cubes_for_data()

        # iterate through the number of cubes and the number of actions
        for i in range(len(cubes.env)):
            for a in range(len(cubes.env[i].action_space)):
                
                

adi(1)