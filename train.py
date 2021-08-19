from DCGAN_core import DCGAN

EPOCH = 30
BATCH_SIZE = 128

if __name__ == '__main__':
    core = DCGAN()
    core.train(epoch=EPOCH, bach_size=BATCH_SIZE)
