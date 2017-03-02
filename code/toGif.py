import imageio
images = []

seriesLength = 11

folder = "../plots/series/"
filenames = [folder+str(x)+".png" for x in range(0,seriesLength)]

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(folder+'interpolation.gif', images, duration=0.5)