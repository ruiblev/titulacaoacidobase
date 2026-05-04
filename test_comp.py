import streamlit as st
import streamlit.components.v1 as components

if "vol" not in st.session_state:
    st.session_state.vol = 0.0

bureta = components.declare_component("bureta", path="bureta_component")
st.write(f"Python volume: {st.session_state.vol:.2f}")

new_v = bureta(vol_add=st.session_state.vol, key="b1")
if new_v is not None and new_v != st.session_state.vol:
    st.session_state.vol = new_v
    st.rerun()
