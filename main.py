import os

for i in range(4,11):
    folder_name = "./threeTwo//L"+str(i)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        continue
    
    
