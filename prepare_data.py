import os


img_path = 'scripts/RAW/JPEGImages/'
img_files = os.listdir(img_path)
txt_path = 'scripts/RAW/labels/'
txt_files = os.listdir(txt_path)
outfile = open('train_darknet.txt','a+')
for img in img_files:
    if img.endswith('.jpg') and img.replace('.jpg','.txt') in txt_files:
        cimpath = os.getcwd()+ '/'+img_path+img
        outfile.write (cimpath + '\n')
        # print (cimpath)

    

# for img in img_files:
#     if not img.replace('.jpg','.txt') in txt_files:
#         os.remove(img_path + img)
