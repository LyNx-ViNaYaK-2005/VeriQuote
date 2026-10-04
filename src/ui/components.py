import streamlit as st

from src.models import Answer


def brand() -> None:
    st.html('<div class="folio-brand"><div class="folio-mark" aria-hidden="true">f.</div>'
            '<span class="folio-wordmark">folio</span></div>')


def empty_state() -> None:
    main, visual = st.columns([1.6, 1], gap="large")
    with main:
        st.html('<div class="folio-eyebrow">Your documents. A clearer perspective.</div>')
        st.title("Read deeper. Ask better.")
        st.html('<p class="folio-subtitle">Turn a stack of PDFs into a focused conversation. '
                'Find the answer, follow the citation, and see the passage behind it.</p>')
        st.caption("PDFs only · No account · A workspace for this session")
    with visual:
        st.html('''<div class="folio-paper" aria-label="Illustration of a document with a highlighted passage">
          <div class="label">THE EVIDENCE COMES WITH IT</div><h3>A little less searching.<br>A little more understanding.</h3>
          <div class="line"></div><div class="line" style="width:78%"></div>
          <div class="highlight">Every answer starts with your pages.</div>
          <div class="line" style="width:90%"></div><div class="line" style="width:62%"></div>
          <div class="foot">READ &nbsp; / &nbsp; QUESTION &nbsp; / &nbsp; VERIFY</div></div>''')


def how_it_works() -> None:
    st.html('<div class="folio-rule"></div>')
    for col, number, title, detail in zip(st.columns(3), ("01", "02", "03"),
        ("Bring your reading", "Ask in your own words", "Follow the evidence"),
        ("Upload text-based PDFs. Folio keeps each passage connected to its page.",
         "Search meaning across your documents, then get a grounded response.",
         "Open the cited passages. Keep a copy of your conversation with Export Chat.")):
        with col:
            st.html(f'<div class="folio-step"><span>{number}</span><strong>{title}</strong><p>{detail}</p></div>')


def render_answer(answer: Answer, active_ids: set[str]) -> None:
    st.markdown(answer.text)
    if answer.note:
        st.caption(answer.note)
    if answer.citations:
        st.caption(f"SUPPORTED BY {len(answer.citations)} SOURCE {'PAGE' if len(answer.citations) == 1 else 'PAGES'}")
    for citation in answer.citations:
        removed = " · removed from workspace" if citation.document_id not in active_ids else ""
        with st.expander(f"[{citation.number}] {citation.source} — Page {citation.page}{removed}"):
            for support in citation.supports:
                st.caption("Supporting excerpt")
                st.text(support.quote)
                st.caption(f"Full retrieved passage · cosine similarity {support.hit.score:.3f}")
                st.text(support.hit.chunk.text)
    if answer.context:
        with st.expander(f"Inspect retrieval · {len(answer.context)} passages considered"):
            st.caption("Similarity measures retrieval proximity, not factual confidence. Only the references above support the answer.")
            for i, hit in enumerate(answer.context, 1):
                st.text(f"E{i} · {hit.chunk.source} · Page {hit.chunk.page}")
                st.caption(f"Cosine similarity {hit.score:.3f}")
                st.text(hit.chunk.text)
