from PIL import Image

datatxt = open("memdump.bin", "rb").read()

#img = Image.frombuffer("RGBA", (1280, 720), datatxt, "raw", "RGBA", 0, 1)
img = Image.frombuffer("RGBA", (1920, 1080), datatxt, "raw", "RGBA", 0, 1)
#print (list(img.getdata())[:6])
img = img.convert('RGB')

img.save("test.PNG")

print ( "done" )