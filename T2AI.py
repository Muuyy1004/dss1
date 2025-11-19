import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from openai import OpenAI

api_key = "sk-U1BAUIuj0Moz0YAYD27e05513c8e44F0AeFf26Bf2bA87b00"
api_base = "https://maas-api.cn-huabei-1.xf-yun.com/v1"
MODEL_ID = "xop3qwen1b7r"
client = OpenAI(api_key=api_key, base_url=api_base)


def ask_ai(messages, json_type=True, model_id=MODEL_ID):
    json_messages = [{"role": "user", "content": messages}]
    if json_type:
        extra_body = {
            "response_format": {"type": "json_object"},
            "search_disable": True
        }
    else:
        extra_body = {}

    response = client.chat.completions.create(
        model=model_id, messages=json_messages, extra_body=extra_body
    )
    message = response.choices[0].message.content
    if json_type:
        message = json.loads(message)
    return message


def ai_explain(task, method, ds_name, highlights):
    prompt = f"""
你是数据科学助教。请用中文简要解读下面的模型结果，并给出3-5条面向管理者的可执行建议（使用•项目符号，不要输出代码）。

任务：{task}；方法：{method}；数据集：{ds_name}
关键结果：{highlights}

请先用1-2句话说明结果意味着什么，再给出建议；
尽量避免术语，聚焦业务含义。
"""
    return ask_ai(prompt, json_type=False)



def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        k_idx = distances.argsort()[:k]
        k_labels = y_train[k_idx]
        pred = np.bincount(k_labels).argmax()
        predictions.append(pred)
    return np.array(predictions)


def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


# ======================================================
#                   STREAMLIT UI
# ======================================================
st.title("📊 决策支持系统")

uploaded = st.file_uploader("上传 CSV 数据（必须包含最后一列为标签）", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("数据预览：", df.head())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    task = "分类"
    ds_name = uploaded.name

    test_ratio = st.slider("测试集比例", 0.1, 0.4, 0.3, 0.05)

    if st.button("训练模型（KNN）"):
        # 手动切分的数据集
        n = len(X)
        split = int(n * (1 - test_ratio))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        y_pred = knn_predict(X_train, y_train, X_test, k=3)

        # 准确率
        acc = (y_pred == y_test).mean()
        st.metric("Accuracy", f"{acc:.3f}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=[f"T_{i}" for i in np.unique(y)],
            columns=[f"P_{i}" for i in np.unique(y)]
        )

        # Altair 热力图
        heat = (
            alt.Chart(cm_df.reset_index().melt("index"))
            .mark_rect()
            .encode(
                x=alt.X("variable:N", title="Pred"),
                y=alt.Y("index:N", title="True"),
                color=alt.Color("value:Q", title="Count")
            )
            .properties(title="Confusion Matrix（SVG）")
        )
        st.altair_chart(heat, use_container_width=True)

        # AI 总结
        highlights = f"Accuracy={acc:.3f}；混淆矩阵规模={cm.shape}。"
        ai_text = ai_explain(task, "KNN（无 sklearn）", ds_name, highlights)
        st.subheader("🤖 AI 解读与管理建议")
        st.write(ai_text)


