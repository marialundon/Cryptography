from cs402 import RSA
import numpy
C=RSA
cipher="mIIyB8an63PKaAEedMbwOgFt-ia"
f=open("cph.txt","w")
f.write(cipher)
f.close()
C.DecipherKey
[7231645985673347207280720222548553948759779729581,4821097323782215625692549251331855329314609896043]
decipher=C.decipher("cph.txt")
print decipher