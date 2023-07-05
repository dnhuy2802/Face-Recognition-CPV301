import { Card, Typography, Space, Button, Popover, Popconfirm } from "antd";
import { MoreOutlined, PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import Flex from "./Flex";
import Spacer from "./Spacer";
import { appStrings } from "../utils/appStrings";

function FaceCard({ id, imgUrl, name, onDelete, onAddMore }) {
  return (
    <Card cover={<img alt="" src={imgUrl} />} size="small">
      <Flex align="center" justify="space-between">
        <Space direction="vertical" size={1}>
          <Typography.Text strong>{name}</Typography.Text>
          <Typography.Text type="secondary">{id}</Typography.Text>
        </Space>
        <Popover
          content={
            <Flex direction="column" align="stretch">
              <Button icon={<PlusOutlined />} onClick={onAddMore}>
                {appStrings.upload.faceCardAddMoreButton}
              </Button>
              <Spacer size={5} />
              <Popconfirm
                title={appStrings.upload.faceCardDeleteConfirmTitle}
                description={appStrings.upload.faceCardDeleteConfirmContent}
                onConfirm={onDelete}
              >
                <Button danger icon={<DeleteOutlined />}>
                  {appStrings.upload.faceCardDeleteButton}
                </Button>
              </Popconfirm>
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
