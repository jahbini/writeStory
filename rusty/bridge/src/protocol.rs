use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub id: String,
    pub cmd: String,
    #[serde(default)]
    pub args: Value,
}

impl Request {
    pub fn validate(&self) -> Result<(), ErrorPayload> {
        if self.id.trim().is_empty() {
            return Err(ErrorPayload {
                code: "bad_request".to_string(),
                message: "id must be a non-empty string".to_string(),
            });
        }
        if self.cmd.trim().is_empty() {
            return Err(ErrorPayload {
                code: "bad_request".to_string(),
                message: "cmd must be a non-empty string".to_string(),
            });
        }
        if !self.args.is_object() {
            return Err(ErrorPayload {
                code: "bad_request".to_string(),
                message: "args must be an object".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorPayload {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub id: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorPayload>,
}

impl Response {
    pub fn ok(id: impl Into<String>, value: Value) -> Self {
        Self {
            id: id.into(),
            ok: true,
            value: Some(value),
            error: None,
        }
    }

    pub fn err(
        id: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            ok: false,
            value: None,
            error: Some(ErrorPayload {
                code: code.into(),
                message: message.into(),
            }),
        }
    }
}

pub fn alive_value() -> Value {
    json!({ "status": "alive" })
}
