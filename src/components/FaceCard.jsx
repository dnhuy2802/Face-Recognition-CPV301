import { Card, Typography, Space, Button, Popover } from "antd";
import { MoreOutlined, PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import Flex from "./Flex";
import Spacer from "./Spacer";

function FaceCard({ id, imgUrl, name }) {
  return (
    <Card cover={<img alt="" src={imgUrl} />} size="small">
      <Flex align="center" justify="space-between">
        <Space direction="vertical" size={1}>
          <Typography.Text strong>{name}</Typography.Text>
          <Typography.Text type="secondary">{id}</Typography.Text>
        </Space>
        <Popover
          content={
            <Flex direction="column">
              <Button icon={<PlusOutlined />}>Add More Images</Button>
              <Spacer size={5} />
              <Button danger icon={<DeleteOutlined />}>
                Delete
              </Button>
            </Flex>
          }
          trigger="click"
        >
          <Button icon={<MoreOutlined />} type="ghost"></Button>
        </Popover>
      </Flex>
    </Card>
  );
}

export default FaceCard;
