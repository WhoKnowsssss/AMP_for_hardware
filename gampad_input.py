import time

from legged_gym.utils import Gamepad, Hand


def main():
    gamepad = Gamepad(0)

    gamepad.start()

    try:
        while True:
            print(gamepad.getX(Hand.left), gamepad.getButtonA())
            time.sleep(0.02)
    except KeyboardInterrupt:
        gamepad.stop()


if __name__ == "__main__":
    main()