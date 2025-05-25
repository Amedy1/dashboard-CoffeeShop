import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- Données ---
df = pd.read_excel("CoffeeShopSales.xlsx", parse_dates=["transaction_date"])
df["CA"] = df["transaction_qty"] * df["unit_price"]
df["heure"] = pd.to_datetime(df["transaction_time"], format='%H:%M:%S').dt.hour
df["jour_semaine"] = df["transaction_date"].dt.day_name()
df["mois"] = df["transaction_date"].dt.to_period("M").astype(str)

# --- App Dash ---
app = Dash(__name__)
app.title = "Dashboard Ventes Cafétéria"

# --- Layout ---
app.layout = html.Div([
    html.H1("Dashboard Ventes Cafétéria", style={"textAlign": "center"}),

    html.Div([
        html.Label("Filtrer par magasin:"),
        dcc.Dropdown(df['store_location'].unique(), multi=True, id='store-filter',
                     value=df['store_location'].unique()),
        html.Label("Filtrer par mois:"),
        dcc.Dropdown(df['mois'].unique(), multi=True, id='mois-filter',
                     value=df['mois'].unique())
    ], style={'padding': '10px 30px'}),

    html.Div(id="kpi-cards", style={"display": "flex", "justifyContent": "space-around", "margin": "20px"}),

    dcc.Graph(id='ca_mensuel'),
    dcc.Graph(id='ca_jour'),
    dcc.Graph(id='heatmap'),
    dcc.Graph(id='top_produits'),

    html.H3("Analyse de panier : Produits achetés ensemble"),
    html.Div(id="panier_table"),
])

# --- Callbacks ---
@app.callback(
    Output('kpi-cards', 'children'),
    Output('ca_mensuel', 'figure'),
    Output('ca_jour', 'figure'),
    Output('heatmap', 'figure'),
    Output('top_produits', 'figure'),
    Output('panier_table', 'children'),
    Input('store-filter', 'value'),
    Input('mois-filter', 'value')
)
def update_dashboard(store_selection, mois_selection):
    df_filtered = df[df['store_location'].isin(store_selection) & df['mois'].isin(mois_selection)]

    # KPIs
    total_ca = df_filtered["CA"].sum()
    nb_transac = df_filtered["transaction_id"].nunique()
    ticket_moyen = total_ca / nb_transac if nb_transac > 0 else 0
    total_qty = df_filtered["transaction_qty"].sum()

    kpis = [
        html.Div(f"CA Total: {total_ca:,.0f} €", style={'fontSize': 20}),
        html.Div(f"Transactions: {nb_transac}", style={'fontSize': 20}),
        html.Div(f"Ticket Moyen: {ticket_moyen:.2f} €", style={'fontSize': 20}),
        html.Div(f"Qté vendue: {total_qty}", style={'fontSize': 20}),
    ]

    # CA mensuel
    ca_mois = df_filtered.groupby("mois")["CA"].sum().reset_index()
    fig_mois = px.line(ca_mois, x="mois", y="CA", title="Évolution du CA par mois", markers=True)

    # CA par jour de la semaine
    jour_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ca_jour = df_filtered.groupby("jour_semaine")["CA"].sum().reindex(jour_order).reset_index()
    fig_jour = px.bar(ca_jour, x="jour_semaine", y="CA", title="CA par jour de la semaine")

    # Heatmap heure x jour
    pivot = df_filtered.pivot_table(index="jour_semaine", columns="heure", values="CA", aggfunc="sum").fillna(0).reindex(jour_order)
    fig_heatmap = px.imshow(pivot, labels=dict(color="CA (€)"), title="Heatmap : Heure x Jour")

    # Top 10 produits
    top = df_filtered.groupby("product_detail")["CA"].sum().nlargest(10).reset_index()
    fig_top = px.bar(top, x="CA", y="product_detail", orientation="h", title="Top 10 produits")

    # Analyse panier
    transactions = df_filtered.groupby("transaction_id")["product_detail"].apply(list)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)
    frequent = apriori(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent, metric="lift", min_threshold=1.2)

    if not rules.empty:
        rules["produits"] = rules["antecedents"].apply(lambda x: ', '.join(x)) + " ➜ " + rules["consequents"].apply(lambda x: ', '.join(x))
        table = rules[["produits", "support", "confidence", "lift"]].round(3).head(10).to_html(index=False)
        panier_table = html.Div([
            html.H5("Top associations de produits :"),
            html.Div(dcc.Markdown(table, dangerously_allow_html=True))
        ])
    else:
        panier_table = html.Div("Aucune règle trouvée.")

    return kpis, fig_mois, fig_jour, fig_heatmap, fig_top, panier_table

# --- Exécution ---
if __name__ == "__main__":
    app.run(debug=True,port=8050, host='0.0.0.0')
