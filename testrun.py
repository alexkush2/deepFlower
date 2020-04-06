from utility import show_image
from flowerModel import FlowerModel

a=FlowerModel()
a.loadModel()
a.pre_processing()
#a.train(20)
# a.saveModel()
i = a.predict("./data/sunflower/24459548_27a783feda.jpg")
print(i)
if i == 0:
    print("daisy")
elif i == 1:
    print("dandelion")
elif i == 2:
    print("rose")
elif i == 3:
    print("sunflower")
elif i == 4:
    print("tulip")
else:
    print("thats not a supported flower!")