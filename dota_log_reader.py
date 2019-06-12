import socket

IP = '127.0.0.1'
port = 3000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP, port))

print("UDP listener opened!")

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print ("received message: " + data.decode("utf-8") )