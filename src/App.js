import React, { useState } from "react";
import { Button, Col, Input, Layout, Row, Form, Radio } from "antd";
import "./App.css";
import { PassageTextInput } from "./components/PassageTextInput/PassageTextInput";
import { DisplayPassage } from "./components/DisplayPassage/DisplayPassage";

export const App = () => {
  const [inputPassageMode, setInputPassageMode] = useState(true);
  const [passage, setPassage] = useState("");

  const formItem = (x) => {
    const array = [];
    for (let i = 1; i <= x; i++) {
      array.push(
        <Form.Item label={`Question ${i}`}>
          <Input placeholder="ANSWER BOX" />
          <Button type="primary">Submit</Button>
        </Form.Item>
      );
    }
    return array;
  };

  return (
    <Layout>
      <Layout.Header />
      <Layout.Content style={{ padding: "16px 50px" }}>
        <Row
          gutter={24}
          style={{
            background: "white",
            padding: 24,
            minHeight: "calc(100vh - 64px - 32px)",
          }}
        >
          <Col span={12}>
            <Radio.Group
              size="large"
              onChange={() => setInputPassageMode(!inputPassageMode)}
              buttonStyle="solid"
              defaultValue="insertPassage"
              style={{ width: "100%", paddingBottom: "16px" }}
            >
              <Radio.Button
                value="insertPassage"
                style={{ width: "50%", textAlign: "center" }}
              >
                Insert Passage
              </Radio.Button>
              <Radio.Button
                value="currentPassage"
                style={{ width: "50%", textAlign: "center" }}
              >
                Current Passage
              </Radio.Button>
            </Radio.Group>
            {inputPassageMode ? (
              <PassageTextInput setPassage={setPassage} />
            ) : (
              <DisplayPassage passage={passage} />
            )}
          </Col>
          <Col span={12}>
            <Form layout="vertical">{formItem(6)}</Form>
          </Col>
        </Row>
      </Layout.Content>
    </Layout>
  );
};

export default App;
