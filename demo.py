from rembg import remove
from PIL import Image
input='D:/New folder/resume/rajuphoto.jpeg'
output='D:/New folder/resume/output.png'
inp = Image.open(input)
out=remove(inp)
out.save(output)