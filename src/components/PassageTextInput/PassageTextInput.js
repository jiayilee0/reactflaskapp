import React from "react";
import { Button, Form, Input } from "antd";
import "./styles.css";

export const PassageTextInput = ({ setPassage }) => {
  const [form] = Form.useForm();

  const onSubmit = ({ textPassage }) => {
    setPassage(textPassage);
    // TODO: send textPassage to API
    form.resetFields();
  };

  return (
    <Form form={form} layout="vertical" onFinish={onSubmit}>
      <Form.Item
        label="Text Passage"
        name="textPassage"
        rules={[{ required: true, message: "Please enter a passage" }]}
      >
        <Input.TextArea
          showCount
          autoSize={{ minRows: 18 }}
          placeholder="Insert passage here"
        />
      </Form.Item>
      <Form.Item>
        <Button
          type="primary"
          htmlType="submit"
          style={{ margin: "16px 0", width: "100%" }}
        >
          Submit
        </Button>
      </Form.Item>
    </Form>
  );
};
