"""A tiny native Streamlit v2 component; no iframe or Node build needed."""
import streamlit as st


_guard = st.components.v2.component(
    "folio_session_guard",
    js="""
    export default function(component) {
        // Render runs again on data changes; cleanup is only guaranteed on
        // unmount. Remove the previous listener explicitly on every render.
        const key = Symbol.for('folio.beforeunload');
        if (window[key]) {
            window.removeEventListener('beforeunload', window[key]);
            delete window[key];
        }
        const guard = (event) => {
            event.preventDefault();
            event.returnValue = '';
        };
        if (component.data.active) {
            window[key] = guard;
            window.addEventListener('beforeunload', guard);
        }
        return () => {
            window.removeEventListener('beforeunload', guard);
            if (window[key] === guard) delete window[key];
        };
    }
    """,
)


def protect_session(active: bool) -> None:
    # The browser decides whether to display its generic confirmation.
    _guard(data={"active": active}, key="session_loss_guard")
