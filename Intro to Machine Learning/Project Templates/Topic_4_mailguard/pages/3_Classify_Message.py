"""Page 3: Classify a message and explain the decision.

Paste any text; the filter returns the probability it is spam and shows the
words that most strongly indicate spam, so the decision is transparent.
"""

import streamlit as st

from src.model import classify_message, influential_words
from src.ui_data import get_trained_filter

st.title("3. Classify a Message")
st.write(
    "Type or paste a message below. The filter scores it and explains itself by "
    "listing the most spam-indicative words it learned during training."
)

pipe, _, _ = get_trained_filter()

default = "Congratulations! Click now to claim your free cash prize, limited offer!"
text = st.text_area("Message text", value=default, height=120)

if st.button("Classify"):
    prob = classify_message(pipe, text)
    st.subheader("Result")
    st.metric("Spam probability", f"{prob:.1%}")
    if prob >= 0.5:
        st.error("This message looks like **spam**.")
    else:
        st.success("This message looks like legitimate mail (**ham**).")
    st.progress(prob)

st.subheader("Words the model finds most spam-indicative")
st.write(
    "These are the words with the strongest lean toward spam across the whole "
    "training corpus (not just your message)."
)
st.dataframe(influential_words(pipe, top_n=10), use_container_width=True)

st.caption(
    "The score multiplies the evidence from every word. Even neutral filler "
    "words carry a little weight — that is naive Bayes weighing all the clues."
)
