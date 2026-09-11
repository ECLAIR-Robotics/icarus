import rclpy
from rclpy.node import Node


class IcarusNode(Node):
    def __init__(self):
        super().__init__('icarus_node')
        self.get_logger().info('icarus_node started')


def main(args=None):
    rclpy.init(args=args)
    node = IcarusNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
