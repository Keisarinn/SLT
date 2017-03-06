from pylab import imshow, show, cm


#def view_image(image, label=""):
#    """View a single image."""
#    print("Label: %s" % label)
#    imshow(image, cmap=cm.gray)
#    show()

def view_image(image):
    """View a single image."""
    imshow(image, cmap=cm.gray)
    show()
