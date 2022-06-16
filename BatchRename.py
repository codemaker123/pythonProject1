import os
import shutil

class BatchRename():

    def __init__(self):
        # 需要处理的文件夹
        self.path = 'D:\PyCharm\Projects\PythonProject\Pics'
        # 保存重命名后的图片地址
        self.save_path = 'D:\PyCharm\Projects\PythonProject\Pics1'

    def rename(self):
        #获取文件路径
        filelist = os.listdir(self.path)
        #获取文件长度（个数）
        total_num = len(filelist)
        i = 1 #文件命名从1开始
        for item in filelist:
            print(item)
            if item.endswith('.bmp'): #初始图片格式未bmp
                src = os.path.join(os.path.abspath(self.path),item)  #当前文件中图片的地址
                dst = os.path.join(os.path.abspath(self.save_path), ''+str(i) + '.jpg') #处理后的图片名称
                try:
                    shutil.copy(src,dst)
                    #os.rename(src,dst)
                    print('converting %s to %s ...' % (src,dst))
                    i = i +1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num,i))