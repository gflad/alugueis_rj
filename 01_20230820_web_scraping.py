
# bibliotecas
from concurrent.futures import ThreadPoolExecutor
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd


# %% pegando os links de cada anuncio
# link da pesquisa
link = "https://www.olx.com.br/imoveis/estado-rj/rio-de-janeiro-e-regiao?o=1&q=aluguel"

# definindo o cabecalho para o site nao entender que estamos acessando atraves do python
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203"}

# fazendo a requisicao de acesso ao site com a biblioteca 'requests'
requisicao = requests.get(link, headers=headers)

# se a requisicao der certo, vai retornar o valor 200:
print(requisicao)

# printar o 'requisicao.text' traz muita informacao desorganizada
# print(requisicao.text)

# solucao: BeautifulSoup
site = BeautifulSoup(requisicao.text, "html.parser")
print(site.prettify())

# encontrando links dos anuncios
# class="sc-1mburcf-1 hqJEoJ"
# pegando o link de cada anuncio individual
pesquisa_links = site.find_all("a", class_="etGiBL")
links = [link.get('href') for link in pesquisa_links]

print(links)

# printando de forma a visualizar melhor
for link in links:
    print(link)


# %% automatizando para fazer o mesmo procedimento com n paginas da olx:

# lista vazia para armazenar todos os links
all_links = []

# inicializa o timer
start_time = time.time()

# loop pelas primeiras 100 paginas
for page_number in range(1, 101):
    links = []

    # roda o loop novamente caso nao retorne link de primeira
    while len(links) == 0:
        # constroi o URL de cada pagina ao mudar o parametro 'o'
        link = f"https://www.olx.com.br/imoveis/estado-rj/rio-de-janeiro-e-regiao?o={page_number}&q=aluguel"

        # definindo headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203"}

        # faz a requisicao
        requisicao = requests.get(link, headers=headers)

        # status da requisicao
        print(requisicao)

        # BeautifulSoup para fazer o parse do HTML
        site = BeautifulSoup(requisicao.text, "html.parser")

        # enconra os links para a pagina atual
        pesquis_links = site.find_all("a", class_="etGiBL")
        links = [link.get('href') for link in pesquis_links]

        print(f"Página {page_number}: {len(links)} links encontrados")

        # pausa de 1 segundo para nao sobrecarregar o servidor
        if len(links) == 0:
            print(f"Tentando página {page_number} novamente")
            import time
            time.sleep(1)  # Pause for 1 second

    # adiciona os links da pagina atual para a lista 'all_links'
    all_links.extend(links)

print(f"Total de links encontrados: {len(all_links)}")

# finaliza o timer
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Esse código levou {elapsed_time} segundos para rodar.")
# Esse código levou 352.5044605731964 segundos para rodar.


# %% fazendo o scraping de cada link resgatado

# funcao para extrair informacoes uteis de cada link
def scrape_link(link):
    for _ in range(3):  # caso de erro, tenta novamente ate 3 vezes
        time.sleep(0.1)  # tempo de pausa para nao sobrecarregar
        try:
            response = requests.get(link, headers=headers)
            response.raise_for_status()  

            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')

            result_dict = {}
            rent_value = soup.find('h2', class_='ad__sc-12l420o-1 dnbBJL sc-VigVT gVrrBf', attrs={'data-testid': None})
            if rent_value:
                result_dict["aluguel"] = rent_value.text.strip()

            result_str = ""
            info_divs = soup.find_all('div', class_='sc-jWBwVP ad__sc-1f2ug0x-3 eQlxUw')
            for info_div in info_divs:
                spans = info_div.find_all('span')
                if len(spans) >= 2:
                    result_str += f'"{spans[0].text.strip()}"; "{spans[1].text.strip()}" '

            result_items = result_str.split('" "')
            for item in result_items:
                key, value = item.strip().split('"; "')
                result_dict[key.replace('"', '').strip()] = value.replace('"', '').strip()

            return result_dict
        except requests.RequestException as e:
            print(f"Erro em retornar {link}: {e}")
            continue  # Retry the loop
    print(f"Falha em retornar {link} depois de 3 tentativas")
    return None


# inicializa o timer
start_time = time.time()

rows = []

# quantos links buscar por vez (para nao sobrecarregar)
tamanho_lote = 500

# iterando os links em lotes de 500
for i in range(0, len(all_links), tamanho_lote):
    batch_links = all_links[i:i + tamanho_lote]
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(scrape_link, batch_links)
        rows.extend(result for result in results if result is not None)

# converte a lista de dicionarios para dataframe
df = pd.DataFrame(rows)

# finaliza o timer
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Esse código levou {elapsed_time} segundos para rodar.")
# Esse código levou 345.09945368766785 segundos para rodar.

# salvando para csv
df.to_csv('web_scraping\\olx_20230820.csv', index=False, encoding='latin-1')

