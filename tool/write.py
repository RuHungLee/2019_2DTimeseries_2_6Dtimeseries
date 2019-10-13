import os
file = os.listdir('./6d_to_2d/2D_point')
file.sort()
f = open('./train_pair.txt' , 'w')
for name in file:
    f.write(f'/home/eric123/2D_to_6D_model/6d_to_2d/2D_point/{name}\t/home/eric123/2D_to_6D_model/6d_to_2d/6D_point/{name}\n')
f.close()
