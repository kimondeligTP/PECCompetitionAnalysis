import numpy as np
import math as math
import pandas as pd
import xml.etree.ElementTree as ET
import io

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import random
import os
import openai
import numpy as np
from transformers import GPT2TokenizerFast
import streamlit as st
from PIL import Image
import time
import pickle
import base64
import requests
import lxml.html as lh
from statistics import mean

import requests
from bs4 import BeautifulSoup

import string


class Account_Data:
    def __init__(self, company_name, package_name):
        self.company_name = company_name  # vector map of each word and their corresponding value set from the tokenizer.word_index
        self.package_name = package_name  # our list of phrases
        self.price_per_kwh_day = 0
        self.price_per_kwh_night = 0
        self.pagio_day = 0
        self.pagio_night = 0
        self.behavioral_discount = 0
        self.discount_pagio = 0
        self.gifts = 0
        self.extra_cost = 0
        self.ritra = False
        self.free_tech_support = True

    def set_kwh_day(self, data):
        self.price_per_kwh_day = data

    def set_kwh_night(self, data):
        self.price_per_kwh_night = data

    def set_pagio_day(self, data):
        self.pagio_day = data

    def set_pagio_night(self, data):
        self.pagio_night = data

    def set_behavioral_discount(self, data):
        self.behavioral_discount = data

    def show_data(self):
        print(self.price_per_kwh_day, self.price_per_kwh_night)


class Del:
    def __init__(self, keep=string.digits):
        self.comp = dict((ord(c), c) for c in keep)

    def __getitem__(self, k):
        return self.comp.get(k)


start_time = time.time()

DD = Del()

plans_list = []

# scrapping values for Home Enter
pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\dei_enter.pkl", mode = "rb")
dei_enter = pickle.load(pickle_in)
plans_list.append(dei_enter)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\dei_enter_plus.pkl", mode = "rb")
dei_enter_plus = pickle.load(pickle_in)
plans_list.append(dei_enter_plus)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\heron_only_home.pkl", mode = "rb")
heron_only_home = pickle.load(pickle_in)
plans_list.append(heron_only_home)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\heron_up.pkl", mode = "rb")
heron_up = pickle.load(pickle_in)
plans_list.append(heron_up)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\heron_night.pkl", mode = "rb")
heron_night = pickle.load(pickle_in)
plans_list.append(heron_night)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\protergia_home.pkl", mode = "rb")
protergia_home = pickle.load(pickle_in)
plans_list.append(protergia_home)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\protergia_charge.pkl", mode = "rb")
protergia_charge = pickle.load(pickle_in)
plans_list.append(protergia_charge)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\protergia_mvp.pkl", mode = "rb")
protergia_mvp = pickle.load(pickle_in)
plans_list.append(protergia_mvp)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\wv_liberty.pkl", mode = "rb")
wv_liberty = pickle.load(pickle_in)
plans_list.append(wv_liberty)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\elpedison_free.pkl", mode = "rb")
elpedison_free = pickle.load(pickle_in)
plans_list.append(elpedison_free)

pickle_in =  open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\volton_flexi_plus.pkl", mode = "rb")
volton_flexi_plus = pickle.load(pickle_in)
plans_list.append(volton_flexi_plus)





@st.cache
def retrieve_dataframe_providers(plans_list):
    provider_name_package = []
    for plan in plans_list:
        provider_name_package.append(plan.package_name)
    provider_table = pd.DataFrame(columns=['Company','Price per kwh day','Price per kwh night', 'Pagio Day',
                                           'Pagio Night','Synepeia Discount','Gifts','Extra',
                                           'Ritra','Free Tech Support','OTS Text'], index=[provider_name_package])

    print(provider_table)
    for plan in plans_list:
        provider_table.loc[plan.package_name] = [plan.company_name,plan.price_per_kwh_day,plan.price_per_kwh_night,
                                                                        plan.pagio_day, plan.pagio_night, plan.behavioral_discount,
                                                                        plan.gifts,plan.extra_cost,plan.ritra,plan.free_tech_support,
                                                                        plan.ots_text]


    return provider_table


def pick_current_provider(provider):
    if provider == 'DEH Home Enter':
        return dei_enter
    else:
        return dei_enter_plus


def pick_competitor(proposed_provider):
    if proposed_provider == 'Protergia MVP':
        return protergia_mvp
    if proposed_provider == 'Protergia Charge':
        return protergia_charge
    if proposed_provider == 'Heron Home Up':
        return heron_up
    if proposed_provider == 'Heron Only Home':
        return heron_only_home
    if proposed_provider == 'W+V Liberty Choice':
        return wv_liberty
    if proposed_provider == 'Elpedison Free':
        return elpedison_free
    if proposed_provider == 'Protergia Home':
        return protergia_home
    if proposed_provider == 'Volton Unique Flexi Plus':
        return volton_flexi_plus
    if proposed_provider == 'Heron Nixterini Xreosi 24/7':
        return heron_night


def breaking_arguements(competitor,our):
    breaking_points = []
    if our.price_per_kwh_day > competitor.price_per_kwh_day:
        point = '{} {} offers  HIGHER price per/kwh day!'.format(
            our.company_name, our.package_name)
        breaking_points.append(point)
    if our.pagio_day > competitor.pagio_day:

        point = '{} {} offers  HIGHER  pagio!'.format(
            our.company_name, our.package_name,)
        breaking_points.append(point)
    return breaking_points

def selling_arguements(competitor, our):
    selling_points = []
    if (our.free_tech_support == True) and (competitor.free_tech_support == False):
        point = 'Το {} προσφέρει δωρεάν υπηρεσία επείγουσας τεχνικής βοήθειας για το σπίτι!'.format(our.package_name)
        selling_points.append(point)
    if our.pagio_day < competitor.pagio_day:
        diff = 100 * ((competitor.pagio_day - our.pagio_day) / our.pagio_day)
        diff = round(diff, 2)
        point = 'Το {} προσφέρει {}% χαμηλότερο πάγιο! (€{} {} vs €{} {})'.format(
             our.package_name, diff, our.pagio_day,our.company_name, competitor.pagio_day,competitor.company_name)
        selling_points.append(point)
    if our.term_of_contract < competitor.term_of_contract:
        point = 'Το {} προσφέρει ευελεξία με {}μηνη διάρκεια σύμβασης, σε σύγκριση με τον πάροχο {}' \
                ' όπου η δέσμευση είναι {}μηνη'.format(our.package_name, our.term_of_contract,competitor.company_name,
                                                       competitor.term_of_contract)
        selling_points.append(point)

    if our.behavioral_discount <= competitor.behavioral_discount:
        point = '***Η {} βασίζει τις καλές τιμές της στην εκπτωση συνέπειας! {}% vs {}%'.format(
            competitor.company_name, our.behavioral_discount * 100, 100 * competitor.behavioral_discount)
        selling_points.append(point)
    selling_points.append('****Δεν απαιτείται εγγύηση-προκαταβολή για την σύναψη σύμβασης για το προϊόν {}'.format(our.package_name))

    if our.price_per_kwh_day < competitor.price_per_kwh_day:
        diff = 100 * ((competitor.price_per_kwh_day - our.price_per_kwh_day) / our.price_per_kwh_day)
        diff = round(diff, 2)
        point = '{} {} offers {}% lower price per/kwh day!(€{} vs €{})'.format(
            our.company_name, our.package_name, diff, our.price_per_kwh_day, competitor.price_per_kwh_day)
        selling_points.append(point)
    if our.pagio_night < competitor.pagio_night:
        diff = 100 * ((competitor.pagio_night - our.pagio_night) / our.pagio_night)
        diff = round(diff, 2)
        point = '{} {} offers {}% lower price pagio night!(€{} vs €{})'.format(
            our.package_name, our.company_name, diff, our.pagio_night, competitor.pagio_night)
        selling_points.append(point)
    if our.extra_cost < competitor.extra_cost:
        point = '{} {} offer €{} extra cost vs current provider €{}'.format(
            our.company_name, our.package_name, our.extra_cost, round(competitor.extra_cost,2))
        selling_points.append(point)
    if our.term_of_contract >= competitor.term_of_contract:
        point = "Το {} προσφέρει σταθερές τιμές καθ' όλη τη διάρκεια της {}μηνης σύμβασης με την ΔΕΗ.".format(
            our.package_name, our.term_of_contract)
        selling_points.append(point)
    return selling_points

def color_negative_red(value):
  """
  Colors elements in a dateframe
  green if positive and red if
  negative. Does not color NaN
  values.
  """

  if value < 0:
    color = 'green'
  elif value > 0:
    color = 'red'
  else:
    color = 'black'

  return 'color: %s' % color





@st.cache(suppress_st_warning=True)
def monthly_behavrioral_cost(plan,Day_kwh_use,night_kwh_use, ritra):
    days_in_a_month = 30
    # Get the table of monthly mwh cost'''
    monthly_mwh = get_mwh_OTS_cost().values[0][-4:]
    four_mo_trail = []
    for months in monthly_mwh:
        c = ritra_coeficient(plan,months / 1000,0.01)
        four_mo_trail.append(c)
    mean_coef = mean(four_mo_trail)
    if ritra == False:
        cost = plan.extra_cost + days_in_a_month * (1 - plan.behavioral_discount) * (
                Day_kwh_use * plan.price_per_kwh_day + night_kwh_use * plan.price_per_kwh_night)\
               + plan.pagio_night + plan.pagio_day
        cost = round(cost, 2)
    else:
        no_r = plan.extra_cost + days_in_a_month * ((1 - plan.behavioral_discount)) * (
                Day_kwh_use * plan.price_per_kwh_day + night_kwh_use * plan.price_per_kwh_night) \
               + plan.pagio_night + plan.pagio_day
        ritra = mean_coef * (night_kwh_use + Day_kwh_use) * days_in_a_month
        cost = no_r + ritra
        cost = round(cost, 2)
    return cost

@st.cache
def monthly_cost(plan,Day_kwh_use,night_kwh_use, ritra):
    days_in_a_month = 30
    monthly_mwh = get_mwh_OTS_cost().values[0][-4:]
    four_mo_trail = []
    for months in monthly_mwh:
        c = ritra_coeficient(plan, months / 1000, 0.01)
        four_mo_trail.append(c)
    mean_coef = mean(four_mo_trail)
    if ritra == False:
        cost = plan.extra_cost + days_in_a_month * (
                Day_kwh_use * plan.price_per_kwh_day + night_kwh_use * plan.price_per_kwh_night)\
               + plan.pagio_night + plan.pagio_day
        cost = round(cost, 2)
    else:
        cost = plan.extra_cost + days_in_a_month *  (
                Day_kwh_use * (mean_coef + plan.price_per_kwh_day) + night_kwh_use * (mean_coef + plan.price_per_kwh_night)) \
               + plan.pagio_night + plan.pagio_day
        cost = round(cost, 2)
    return cost

def convert_period(cost):
    new_cost = cost * 4
    new_cost = round(new_cost,2)
    return new_cost

def misseed_months_overtake(provider,competitor_provider,Day_kwh_use,night_kwh_use):
    our_cost = monthly_cost(provider,Day_kwh_use,night_kwh_use, True)
    provider_cost = monthly_cost(competitor_provider,Day_kwh_use,night_kwh_use, True)

    our_cost_behavior = monthly_behavrioral_cost(provider,Day_kwh_use,night_kwh_use, True)
    provider_cost_behavior = monthly_behavrioral_cost(competitor_provider,Day_kwh_use,night_kwh_use, True)

    if our_cost_behavior < provider_cost_behavior:
        return 0

    for i in range(1,4):
        our_yearly = our_cost_behavior * (12 - (i * 4)) + our_cost * (i * 4)
        provider_yearly = provider_cost_behavior * (12 - (i * 4)) + provider_cost * (i * 4)
        if our_yearly < provider_yearly:
            return i
    return 0

@st.cache
def get_mwh_OTS_cost():
    pickle_in = open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\Scrap\mwh_cost_pickle.pkl",
                     mode="rb")
    mwh_cost_table = pickle.load(pickle_in)
    return mwh_cost_table

@st.cache
def get_monthly_ritra_cost(competitor_provider,Day_kwh_hours, night_kwh_hours):
    mwh_cost_table = get_mwh_OTS_cost()
    ritra_table_monthly = pd.DataFrame(
        columns=[mwh_cost_table.columns],
        index=['Μηνιαίο πληρωτέο ποσό σε ΟΤΣ'])

    ritra_values = []
    for matrix in mwh_cost_table.values:
        for val in matrix:
            kwh = val / 1000
            ritra_coef = ritra_coeficient(competitor_provider, kwh, 0.01)
            ritra_values.append(ritra_coef * (Day_kwh_hours + night_kwh_hours))
    print('Average MonthlyCost', mean(ritra_values))
    ritra_table_monthly.loc[ritra_table_monthly.index] = [ritra_values]

    return ritra_table_monthly




def ritra_coeficient(competitor_provider,market_kwh_cost,added_cost_coef):
    # 0.05695
    protergia_function = 1.15 * market_kwh_cost + 0.0115
    heron_function = 1.16 * market_kwh_cost + 0.0115
    elpedison_function = (market_kwh_cost + added_cost_coef) + (1 + 0.1371) + 0.005
    wv_function = (market_kwh_cost + added_cost_coef) * (1 + 0.1371)
    volton_function = (market_kwh_cost + added_cost_coef) * (1 + 0.1371)

    if competitor_provider.company_name == 'DEH':
        return 0
    elif competitor_provider.company_name == 'Protergia':
        if protergia_function > competitor_provider.ritra_upper_limit:
            return protergia_function - competitor_provider.ritra_upper_limit
        else:
            return protergia_function - competitor_provider.ritra_lower_limit

    elif competitor_provider.company_name == 'Heron':
        if heron_function > competitor_provider.ritra_upper_limit:
            return heron_function - competitor_provider.ritra_upper_limit
        else:
            return heron_function - competitor_provider.ritra_lower_limit

    elif competitor_provider.company_name == 'Elpedison':
        if elpedison_function > competitor_provider.ritra_upper_limit:

            return (market_kwh_cost + added_cost_coef - competitor_provider.ritra_upper_limit) * (1 + 0.1371) + 0.005
        else:
            return (market_kwh_cost + added_cost_coef - competitor_provider.ritra_lower_limit) * (1 + 0.1371) + 0.005

    elif competitor_provider.company_name == 'W+V':
        if wv_function > competitor_provider.ritra_upper_limit:
            return wv_function - competitor_provider.ritra_upper_limit
        else:
            return wv_function - competitor_provider.ritra_lower_limit

    elif competitor_provider.company_name == 'Volton':
        if volton_function > competitor_provider.ritra_upper_limit:
            return volton_function - competitor_provider.ritra_upper_limit
        else:
            return volton_function - competitor_provider.ritra_lower_limit


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


def form_submissions():
    Day_kwh_use = st.number_input(label="ΗΜΕΡΑ: Συνολική 4μηνη Κατανάλωση σε kwh:",
                                  help='Εισάγετε την συνολική 4μηνη κατανάλωση ΗΜΕΡΑΣ του πελάτη,'
                                       ' βάσει του τελευταίου λογαριασμού.')
    night_kwh_use = st.number_input(label="ΝΥΧΤΑ: Συνολική 4μηνη Κατανάλωση σε kwh:",
                                    help='Εισάγετε την συνολική 4μηνη κατανάλωση ΝΥΧΤΑΣ του πελάτη,'
                                         ' βάσει του τελευταίου λογαριασμού.'
                                    )
    competitor_provider = st.selectbox("Τρέχων πάροχος του Καταναλωτή",
                                       ("Protergia MVP", "Protergia Charge", 'Heron Only Home',
                                        'Heron Home Up', 'Heron Nixterini Xreosi 24/7',
                                        'W+V Liberty Choice', 'Elpedison Free',
                                        'Protergia Home', 'Volton Unique Flexi Plus'),
                                       help='Διαλέξτε τον τρέχοντα πάροχο του καταναλωτή'
                                       )
    ## Translate the 4-mo input in daily consumption
    Day_kwh_use = (Day_kwh_use / 4) / 30
    night_kwh_use = (night_kwh_use / 4) / 30

    # pick classes of both competition and provider
    st.write("""<font size="2.4">Προτεινόμενο προϊόν από την ΔΕΗ</font>""", unsafe_allow_html=True)
    col1_1, col2_2 = st.beta_columns(2)
    with col1_1:
        Dei_Home = st.checkbox(label="ΔΕΗ Home Enter", value=False)
    with col2_2:
        Dei_Home_Plus = st.checkbox(label="ΔΕΗ Home Enter+", value=False)
    if Dei_Home:
        provider = "DEH Home Enter"
    else:
        provider = "DEH Home Enter Plus"

    # return the provider from the pickle extract
    provider = pick_current_provider(provider)
    competitor_provider = pick_competitor(competitor_provider)

    return Day_kwh_use,night_kwh_use,competitor_provider,provider,competitor_provider

def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:orange;padding:33px"> 
    <h1 style ="color:white;text-align:center;">Agent Tool</h1> 
    </div> 
    """
    st.set_page_config(layout="wide", initial_sidebar_state='expanded',page_icon="🧊")
    col1, col2, = st.beta_columns([5, 2])



    # display the front end aspect

    # st.markdown(html_temp, unsafe_allow_html=True)
    img = Image.open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\TAP_logo.png.png")
    deh_header = Image.open(r"C:\Users\Deligiannis.7\PycharmProjects\PythonTest\StreamLit\DEH_Header.png")


    #st.markdown(page_bg_img, unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <p class="big-font">
        Using this tool developed for TPGR DEH WinBack campaign we can compare our proposing
        products with the client's existing provider. In order to do so, you should enter the following
        details and the tool will provide you with useful insights: </p>

        <ul>
        <li style="font-size:12px"> Customer Estimated Daily kwh usage </li>
        <li style="font-size:12px"> Customer Estimated Nightly kwh usage </li>
        <li style="font-size:12px"> DEH Product to compare </li>
        <li style="font-size:12px"> Customer's Current Energy Provider </li>
        </ul> 

        <p class="big-font">
        After you enter the details above the tool with calculate <b>monthly</b> average
        expected cost for each product and will also give you selling points to aid with discussion.
        </p>
        """, unsafe_allow_html=True)

    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 947px;
            padding-top: 0rem;
            padding-right: 0.5rem;
            padding-left: 0.5rem;
            padding-bottom: 2rem;
        }}
        .reportview-container .main {{
            color:black;
            background-color: white;
            background-image: img_deh_background;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    

    st.image(deh_header)
    # with col1:
    #     st.write(html_temp,unsafe_allow_html=True)
    # with col2:
    #     st.image(deh_header)
    #     #st.image(img_deh_background)
    st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )
    query_params = st.experimental_get_query_params()
    tabs = ["Ανάλυση Ανταγωνισμού", "Σύγκριση Απώλειας Συνέπειας", "View Data"]
    if "tab" in query_params:
        active_tab = query_params["tab"][0]
    else:
        active_tab = "Tool"

    if active_tab not in tabs:
        st.experimental_set_query_params(tab="Ανάλυση Ανταγωνισμού")
        active_tab = "Ανάλυση Ανταγωνισμού"

    li_items = "".join(
        f"""
        <li class="nav-item">
            <a class="nav-link{' active' if t == active_tab else ''}" href="/?tab={t}">{t}</a>
        </li>
        """
        for t in tabs
    )
    tabs_html = f"""
        <ul class="nav nav-tabs">
        {li_items}
        </ul>
    """
    st.markdown(tabs_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if active_tab == "View Data":
        plans_table = retrieve_dataframe_providers(plans_list)
        st.write(plans_table)

    if active_tab == "Σύγκριση Απώλειας Συνέπειας":
        Day_kwh_use,night_kwh_use,competitor_provider,provider,competitor_provider = form_submissions()


        # Picking the months the customer might have forgot to pay
        st.write(" ")
        st.write("____________________________",unsafe_allow_html=True)
        st.write("Επιλέξτε τους μήνες που ο πελάτης ΔΕΝ ήταν συνεπής στις πληρωμές του τρέχοντος παρόχου. ")
        st.write(" ")

        row1_1, row1_2, row1_3, row1_4 = st.beta_columns((2, 3, 3, 1))
        with row1_1:
            january = st.checkbox('ΙΑΝΟΥΑΡΙΟΣ ')
            st.write(" ")
            may = st.checkbox('ΜΑΪΟΣ ')
            st.write(" ")
            september = st.checkbox('ΣΕΠΤΕΜΒΡΙΟΣ ')
        with row1_2:
            february = st.checkbox('ΦΕΒΡΟΥΑΡΙΟΣ ')
            st.write(" ")
            june = st.checkbox('ΙΟΥΝΙΟΣ ')
            st.write(" ")
            october = st.checkbox('ΟΚΤΩΒΡΙΟΣ')
        with row1_3:
            march = st.checkbox('ΜΑΡΤΙΟΣ ')
            st.write(" ")
            july = st.checkbox('ΙΟΥΛΙΟΣ ')
            st.write(" ")
            november = st.checkbox('ΝΟΕΜΒΡΙΟΣ ')
        with row1_4:
            april = st.checkbox('ΑΠΡΙΛΙΟΣ ')
            st.write(" ")
            august = st.checkbox('ΑΥΓΟΥΣΤΟΣ ')
            st.write(" ")
            december = st.checkbox('ΔΕΚΕΜΒΡΙΟΣ ')

        months_missed = 0

        if january or february or march or april:
            months_missed += 4
        if may or june or july or august:
            months_missed += 4
        if september or october or november or december:
            months_missed += 4

        competitor_yearly_cost = monthly_cost(competitor_provider,Day_kwh_use,night_kwh_use,True) *\
                                 months_missed + monthly_behavrioral_cost(competitor_provider,Day_kwh_use,night_kwh_use,True) * (12-months_missed)
        competitor_yearly_cost = round(competitor_yearly_cost,2)
        our_yearly_cost = monthly_cost(provider, Day_kwh_use, night_kwh_use, True) * \
                                 months_missed + monthly_behavrioral_cost(provider, Day_kwh_use,
                                                                          night_kwh_use, True) * (12 - months_missed)
        our_yearly_cost = round(our_yearly_cost,2)
        st.write(" ")
        st.write(" ")
        st.write("____________________________", unsafe_allow_html=True)

        competitor_normal = monthly_behavrioral_cost(competitor_provider,Day_kwh_use,night_kwh_use,True) * (12)

        competitor_yearly_cost = round(competitor_yearly_cost,2)

        st.write("""<h5 style ="color:black;text-align:left;">Ο πελάτης που δεν είναι συνεπής στην πληρωμή λογαριασμού, χάνει την έκπτωση συνέπειας
                 <strong>{}%</strong> για τους επόμενους 4 μήνες.</h5> 
                            """.format(
            round(competitor_provider.behavioral_discount * 100, 2)),
                 unsafe_allow_html=True)


        st.write(
            'Ετήσιο ποσό που προκύπτει από την απώλεια της {}μηνης έκπτωσης με τον πάροχο {}: <strong>€{}</strong> '.format(
                int(months_missed / 1), competitor_provider.company_name, round(competitor_yearly_cost - competitor_normal, 2)), unsafe_allow_html=True)
        st.write("_____________________________")

        ## ΤΙΤΤΛΕ

        st.write("""<h5 style ="color:black;text-align:left;">Σύγκριση μεταξύ {} κ {} για το ετήσιο
         ποσό που πληρώνει ο πελάτης, εφόσον χάσει την έκπτωση συνέπειας για {} μήνες.</h5> 
                                   """.format(provider.company_name,competitor_provider.company_name,months_missed),
            unsafe_allow_html=True)
        st.write(" ")
        st.write('<strong>{} {}</strong> - Ετήσιο ποσό πληρωμής, εφόσον χάσει την έκπτωση συνέπειας για <strong>{}</strong> μήνες: <strong>€{}</strong>'.format(
            provider.company_name,provider.package_name,months_missed,our_yearly_cost),unsafe_allow_html=True)
        st.write('<strong>{}</strong> - Ετήσιο ποσό πληρωμής, εφόσον χάσει την έκπτωση συνέπειας για <strong>{}</strong> μήνες: <strong>€{}</strong>'.format(
            competitor_provider.company_name, months_missed, competitor_yearly_cost), unsafe_allow_html=True)
        if our_yearly_cost <= competitor_yearly_cost:
            st.write('Όφελος: <strong>€{}</strong> τον χρόνο!'.format(round(competitor_yearly_cost-our_yearly_cost,2)),unsafe_allow_html=True)

        if our_yearly_cost <= competitor_yearly_cost:
            html_temp = """ 
            <div style ="background-color:green;padding:13px"> 
            <h1 style ="color:white;text-align:center;">Πιο Φτηνό</h1> 
            </div> 
            """
        else:
            html_temp = """ 
            <div style ="background-color:red;padding:13px"> 
            <h1 style ="color:white;text-align:center;">Πιο Ακριβό</h1> 
            </div> 
            """
        st.write(html_temp,unsafe_allow_html=True)
        print("--- %s seconds ---" % (time.time() - start_time))
    # following lines create boxes in which user can enter data required to make prediction
    if active_tab == "Ανάλυση Ανταγωνισμού":
        with st.form("my_form"):
            # Standard Form Deployement
            Day_kwh_use, night_kwh_use, competitor_provider, provider, competitor_provider = form_submissions()

            our_cost = monthly_cost(provider,Day_kwh_use,night_kwh_use,False)
            our_cost_behavior = monthly_behavrioral_cost(provider,Day_kwh_use,night_kwh_use,False)

            competitor_cost = round(monthly_cost(competitor_provider,Day_kwh_use,night_kwh_use,False),2)
            competitor_cost_behavior = round(monthly_behavrioral_cost(competitor_provider,Day_kwh_use,night_kwh_use,False),2)

            selling_points = selling_arguements(competitor_provider, provider)

            if st.form_submit_button("Έλεγχος"):
                ## Get the table of historical mwh costs
                st.write(""" <h3 style ="color:#2d73b4;text-align:left;">Ιστορικό Κόστος MWH της ΟΤΣ στην Αγορά</h3>""",
                         unsafe_allow_html=True)
                st.write('ΟΤΣ: Οριακή Τιμή Συστήματος')
                mwh_table = get_mwh_OTS_cost()
                mwh_table = mwh_table.style.format("€{:.1f}")
                st.write(mwh_table)

                ## Get the table of monthly costs customer paying
                txt = """ 
                    <h3 style="color:#2d73b4;">Μηνιαίο Ποσό που προκύπτει από τις Ρυθμιζόμενες Χρεώσεις που πλήρωσε ο πελάτης στην εταιρεία</span>&nbsp;<span style="color:#004b8c;"><strong>{}</strong></h3>
                    """.format(competitor_provider.company_name)

                st.write(txt,
                         unsafe_allow_html=True)

                monthly_ritra_table = get_monthly_ritra_cost(competitor_provider,Day_kwh_use * 30, night_kwh_use * 30)
                display_monthly_ritra = monthly_ritra_table.style.format("€{:.1f}")
                st.write(display_monthly_ritra)

                # Write ritra(ots) cost to the client
                col1,col2 = st.beta_columns([1.2,1])
                with col1:
                    st.write(
                        'Ετήσια επιπρόσθετη χρέωση για ΟΤΣ:  <strong>€{}</strong> !'.format(
                            round(mean(monthly_ritra_table.values[0]) * 12), 2), unsafe_allow_html=True)
                    st.write(
                        "Μηνιαία επιπρόσθετη χρέωση για ΟΤΣ μεταξύ: <strong>€{} - €{}</strong> ".format(
                            round(mean(monthly_ritra_table.values[0]) * 0.92, 2), #ranges of 'good' values
                            round(mean(monthly_ritra_table.values[0]) * 1.07, 2)), unsafe_allow_html=True)
                with col2:
                    st.write(
                        """ <h5 style ="color:black;text-align:center;background-color:#9ac832;padding:17px;">Η χρέωση αυτή δεν υπάρχει στην ΔΕΗ!</h5>""",
                        unsafe_allow_html=True)
                st.write(" ")
                st.write("Oνομασία Κρυφής Χρέωσης {}: {}  ".format(competitor_provider.company_name,competitor_provider.ots_text))

                if provider.package_name == 'Home Enter Plus':
                    ots_dei = """ <h6 style ="color:red;text-align:left;">Όλοι οι πάροχοι έχουν το δικαίωμα εφόσον η τιμή που αγοράζουν την kwh αυξηθεί, να μετακυλήσουν αυτό το κόστος στον λογαριασμό που καλείται να πληρώσει ο πελάτης. Η ΔΕΗ με τη δημιουργία του προγράμματος myHomeEnter+ καταργεί αυτή τη χρέωση στα τιμολόγια των πελατών της.</h6>"""
                    st.write(ots_dei,unsafe_allow_html=True)
                st.write('')
                st.write("____________________________________________________________")

                ## 4-mo breakdown of costs

                txt = """ <h3 style="color:#2d73b4;">Σύγκριση Κόστους <strong>4μηνης</strong> Κατανάλωσης μεταξύ: </span>&nbsp;<span style="color:#004b8c;"><strong>{}</strong>&nbsp;<span style="color:#004b8c;"> και &nbsp;<span style="color:#004b8c;"><strong>{}</strong></h3>
                                  """.format(provider.company_name, competitor_provider.company_name)

                st.write(txt,
                          unsafe_allow_html=True)


                ritra_table = pd.DataFrame(
                    columns=[provider.company_name, competitor_provider.company_name , '% Διαφορά',
                             'Σε €' ],
                    index=['με συνέπεια', 'χωρίς συνέπεια'])

                competitor_cost_behavior_ritra = monthly_behavrioral_cost(competitor_provider,
                                                                          Day_kwh_use, night_kwh_use, True)

                competitor_cost_ritra = monthly_cost(competitor_provider, Day_kwh_use, night_kwh_use, True)


                synepeia_difference = ((convert_period(our_cost_behavior) - convert_period(competitor_cost_behavior_ritra)) / convert_period(our_cost_behavior))
                cost_difference = ((convert_period(our_cost) - convert_period(competitor_cost_ritra)) / convert_period(our_cost))

                ritra_table.loc['με συνέπεια'] = ['€ ' + str(convert_period(our_cost_behavior)),
                                                  '€ ' + str(convert_period(competitor_cost_behavior_ritra)),synepeia_difference,
                                                  convert_period(our_cost_behavior) - convert_period(competitor_cost_behavior_ritra)]

                ritra_table.loc['χωρίς συνέπεια'] = ['€ ' + str(convert_period(our_cost)), '€ ' + str(convert_period(competitor_cost_ritra)),cost_difference,
                                                     convert_period(our_cost) - convert_period(competitor_cost_ritra)
                                                     ]


                ritra_table = (ritra_table.style
                    .applymap(color_negative_red, subset=['% Διαφορά','Σε €']).format(
                    {'% Διαφορά': "{:,.1%}",'Σε €': "€{:.1f}"}))

                #_____________________
                # Calc Table for diff
                dif_table = pd.DataFrame(
                    columns=[ 'Μηνιαία Διαφορά', 'Ετήσια Διαφορά'],
                    index=['.', '-'])


                dif_table.loc['.'] = [
                                                  round((convert_period(our_cost_behavior) - convert_period(
                                                      competitor_cost_behavior_ritra)) / 4,2),
                                                  round((convert_period(our_cost_behavior) - convert_period(
                                                      competitor_cost_behavior_ritra)) * 3,2)
                                                  ]
                dif_table.loc['-'] = [
                                                     round((convert_period(our_cost) - convert_period(
                                                         competitor_cost_ritra)) / 4,2),
                                                     round(((convert_period(our_cost) - convert_period(
                                                         competitor_cost_ritra)) * 3),2)
                                                     ]
                dif_table = (dif_table.style
                    .applymap(color_negative_red, subset=['Μηνιαία Διαφορά', 'Ετήσια Διαφορά']).format(
                    {'Μηνιαία Διαφορά': "€{:.1f}",'Ετήσια Διαφορά': "€{:.1f}"}))
                #______________________
                st.write("Κόστος 4μηνης κατανάλωσης με ΟΤΣ")
                col_one,col_two = st.beta_columns([1.67,1])
                with col_one:
                    st.write(ritra_table)
                with col_two:
                    st.write(dif_table)

                # Identifying how many months a customer has to miss for DEH to be better

                break_point_months = misseed_months_overtake(provider,competitor_provider,Day_kwh_use,night_kwh_use)
                st.write(" ")
                st.write("____________________________________________________________")
                if break_point_months > 1:
                    st.write('Αν ο πελάτης δεν πληρώσει εμπρόθεσμα <strong>{}</strong> λογαριασμούς του υπάρχοντος παρόχου, τότε η ετήσια χρέωση από την ΔΕΗ είναι πιο οικονομική.!'.format(
                        break_point_months),unsafe_allow_html=True)
                if break_point_months == 1:
                    st.write(
                        'Αν ο πελάτης δεν πληρώσει εμπρόθεσμα <strong>{}</strong> λογαριασμό του υπάρχοντος παρόχου, τότε η ετήσια χρέωση από την ΔΕΗ είναι πιο οικονομική!'.format(
                            break_point_months), unsafe_allow_html=True)
                if break_point_months == 0:
                    st.write(
                        '<strong> Λόγω της ύπαρξης κρυφής ρήτρας (ΟΤΣ) στον υπάρχοντα πάροχο, η ΔΕΗ είναι πιο οικονομική ανεξαρτήτως συνέπειας! </strong>', unsafe_allow_html=True)

                # Option to show monthly consumption
                st.write("____________________________________________________________")
                st.write(" ")
                show_monthly = st.checkbox(label="Μηνιαία ανάλυση:",value=False,help='Μηνιαία ανάλυση')

                if show_monthly:
                    txt = """ <h3 style="color:#2d73b4;">Σύγκριση Κόστους <strong>Μηνιαίας</strong> Κατανάλωσης μεταξύ: </span>&nbsp;<span style="color:#004b8c;"><strong>{}</strong>&nbsp;<span style="color:#004b8c;"> και &nbsp;<span style="color:#004b8c;"><strong>{}</strong></h3>
                                                      """.format(provider.company_name,
                                                                 competitor_provider.company_name)
                    st.write(txt,
                             unsafe_allow_html=True)



    # Ritra
                    ritra_table = pd.DataFrame(
                        columns=[provider.company_name, str(competitor_provider.company_name), 'Διαφορά %'],
                        index=['με συνέπεια', 'χωρίς συνέπεια'])



                    competitor_cost_behavior_ritra = monthly_behavrioral_cost(competitor_provider,
                                                                        Day_kwh_use,night_kwh_use,True)


                    competitor_cost_ritra = monthly_cost(competitor_provider,Day_kwh_use,night_kwh_use,True)

                    synepeia_difference = (our_cost_behavior - competitor_cost_behavior_ritra) / our_cost_behavior

                    cost_difference = (our_cost - competitor_cost_ritra) / our_cost

                    ritra_table.loc['με συνέπεια'] = ['€ ' + str(our_cost_behavior), '€ ' + str(competitor_cost_behavior_ritra),
                                                        synepeia_difference]
                    ritra_table.loc['χωρίς συνέπεια'] = ['€ ' + str(our_cost), '€ ' + str(competitor_cost_ritra),
                                                           cost_difference]

                    ritra_table = (ritra_table.style
                                     .applymap(color_negative_red, subset=['Διαφορά %']).format(
                        {'Διαφορά %': "{:.1%}"}))




                    st.write('Μέσο κόστος μηνιαίας κατανάλωσης με ΟΤΣ',ritra_table)

                st.header('Πλεονεκτήματα του παρόχου ΔΕΗ')
                for i in selling_points:
                    st.success(i)
                st.header('Breaking Points')
                breaking_points = breaking_arguements(competitor_provider, provider)
                for i in breaking_points:
                    html_temp = """ 
                    <div style ="background-color:red;padding:3px"> 
                    <h8 style ="color:white;text-align:center;">"""+i+"""</h8> 
                    </div> 
                    """
                    st.write(html_temp,unsafe_allow_html=True)
                print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
