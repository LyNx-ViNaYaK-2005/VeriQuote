"""Original, restrained document-workspace styling; native widgets stay accessible."""
import streamlit as st


def apply_styles() -> None:
    st.html("""<style>
    :root { --folio-accent: #b9d58b; --folio-muted: #a1ada5; }
    .stApp { background: #101614; }
    [data-testid="stHeader"] { background: #101614ee; }
    [data-testid="stMainBlockContainer"] { max-width: 1150px; padding-top: 5rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #151d19; border-right: 1px solid #2b3730; }
    [data-testid="stSidebarUserContent"] { padding-top: 2rem; }
    h1, h2, h3 { font-weight: 500 !important; letter-spacing: -.035em; }
    h1 { font-size: 3.2rem !important; line-height: 1.08 !important; }
    h2 { font-size: 1.65rem !important; }
    p, li { line-height: 1.65; }
    .folio-brand { display: flex; align-items: center; gap: 12px; margin: 0 0 36px; }
    .folio-mark { display: grid; place-items: center; width: 35px; height: 40px; color: #111913;
      background: #b9d58b; font: 26px Georgia, serif; border-radius: 2px 10px 2px 2px; }
    .folio-wordmark { font: 29px Georgia, serif; letter-spacing: -.04em; color: #eeefe6; }
    .folio-eyebrow { font: 11px ui-monospace, monospace; letter-spacing: .17em; text-transform: uppercase;
      color: #b9d58b; margin-bottom: 16px; }
    .folio-subtitle { color: #a1ada5; font-size: 1.08rem; max-width: 540px; margin: 18px 0 28px; }
    .folio-topline { display: flex; justify-content: space-between; gap: 20px; padding-bottom: 18px;
      border-bottom: 1px solid #2b3730; color: #a1ada5; font-size: 12px; margin-bottom: 28px; }
    .folio-status { color: #b9d58b; }
    .folio-rule { border-top: 1px solid #2b3730; margin: 28px 0; }
    .folio-step { padding: 14px 0; border-top: 1px solid #344137; }
    .folio-step span { color: #b9d58b; font: 12px ui-monospace, monospace; margin-right: 18px; }
    .folio-step strong { font-weight: 500; color: #eeefe6; }
    .folio-step p { margin: 8px 0 0 37px; color: #a1ada5; font-size: 13px; }
    .folio-paper { background: #e6e5d9; color: #253b30; padding: 28px; border-radius: 2px;
      border-top: 4px solid #b9d58b; transform: rotate(2deg); margin: 10px 16px 30px 28px; }
    .folio-paper .label { font: 10px ui-monospace, monospace; letter-spacing: .15em; color: #4e6355; }
    .folio-paper h3 { font: 27px Georgia, serif !important; color: #253b30; margin: 19px 0; }
    .folio-paper .line { height: 5px; background: #c4cabb; margin: 13px 0; }
    .folio-paper .highlight { background: #c5d7a8; padding: 8px 10px; font: 14px Georgia, serif; }
    .folio-paper .foot { border-top: 1px solid #b7c0ae; margin-top: 25px; padding-top: 12px;
      font: 10px ui-monospace, monospace; color: #4e6355; }
    [data-testid="stFileUploader"] section { border: 1px dashed #516346; border-radius: 5px; background: #19221b; }
    [data-testid="stChatMessage"] { background: #17201b; border: 1px solid #2b3730; border-radius: 5px; padding: 1.4rem; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: transparent; border-color: transparent; }
    [data-testid="stMetricValue"] { font-family: ui-monospace, monospace; font-size: 1.6rem; }
    [data-testid="stMetricLabel"] { color: #a1ada5; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #a1ada5; opacity: 1; }
    [data-testid="stExpander"] { border-radius: 4px; }
    .stButton button, .stDownloadButton button { border-radius: 4px; transition: border-color .15s, background .15s; }
    .stButton button:hover, .stDownloadButton button:hover { border-color: #b9d58b; }
    button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 2px solid #c5dfa0 !important; outline-offset: 3px; }
    .st-key-session_loss_guard { display: none; }
    @media(max-width: 760px) {
      [data-testid="stMainBlockContainer"] { padding: 4.5rem 1.1rem 2rem; }
      h1 { font-size: 2.35rem !important; }
      .folio-paper { display: none; }
      .folio-topline { margin-bottom: 15px; }
      [data-testid="stChatMessage"] { padding: 1rem; }
    }
    @media(prefers-reduced-motion: reduce) { * { transition: none !important; } }
    </style>""")
