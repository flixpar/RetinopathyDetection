import glob
import os
import cv2
import tqdm

def main():
    imgs = glob.glob("/home/flixpar/data/retina/*.tif")
    for img_fn in tqdm.tqdm(imgs):
        img = cv2.imread(img_fn)
        if img is None: continue
        img = cv2.resize(img, (1024,1024))
        fn = img_fn.split('/')[-1][:-4]
        cv2.imwrite(f"/home/flixpar/data/imgs/{fn}.png", img)

if __name__ == "__main__":
    main()
