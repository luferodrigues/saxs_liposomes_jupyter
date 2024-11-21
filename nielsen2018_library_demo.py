# Importa bibliotecas
import numpy as np
from scipy.stats import norm
from scipy import special
from lmfit import Minimizer, Parameters

class Liposome_Symmetric():
    def __init__(self, q = [], i_q = [], err_q = [], **kwargs):
        self.z_array = np.linspace(-40, 40, 200)
        self.n_avogadro = 6.022E23
        self.e_radius = 2.818E-5 # in Angstrom
        self.v_water = 29.7 # in Angstrom^3
        self.water_concentration = 55.5 # in M
        self.lipid_density = 1.1E-24
        self.sample_size = 200
        self.r_array = np.zeros(self.sample_size)
        self.distribution_array = np.zeros(self.sample_size)
        self.norm = 1
        self.params = Parameters()
        self.parvals = self.params.valuesdict() # é o que será usado nas funções
        self.features = {
            'distribution': 'gaussian', # Pode ser gaussian, lognormal, schulz ou delta para distribuição de r
            'which': None, # Pode ser p(q), f(q) ou kucerka, descreve a expressão da casca em schulz analítico
            'ch2': 'nielsen' # Pode ser nielsen ou katsaras para descrever a região hidrofóbica da membrana
                        }
        self.volume_distributions = {'p_hg_i' : np.zeros(len(self.z_array)), 
                                     'p_hg_o' : np.zeros(len(self.z_array)), 
                                     'p_hg' : np.zeros(len(self.z_array)),
                                     'p_cg_i' : np.zeros(len(self.z_array)),
                                     'p_cg_o' : np.zeros(len(self.z_array)),
                                     'p_cg' : np.zeros(len(self.z_array)),
                                     'p_hc' : np.zeros(len(self.z_array)),
                                     'p_ch2' : np.zeros(len(self.z_array)),
                                     'p_ch3' : np.zeros(len(self.z_array)),
                                     'p_p' : np.zeros(len(self.z_array)),
                                     'p_water' : np.zeros(len(self.z_array)),
                                     'p_total' : np.zeros(len(self.z_array)),
                                     }
        
    # Cria objeto Parameter do LMFIT. Os kwargs podem ser value, min, max e expr, por exemplo.
    def add_param(self, name, **kwargs):
        self.params.add(name, **kwargs)
        self.update_params()
        
    def replace_params(self, lmfit_params):
        for item in lmfit_params:
            self.params.add(item.name)
    
    def get_params_from_fit(self, resultado):
        dictionary = {name:params.value for name, params in resultado.params.items()}
        return dictionary
        # O QUE TÁ ACONTECENDO AQUI EM CIMA
        # dictionary = {} 
        # for item in resultado.params.items():
            # dictionary.update(item: resultado.params.items()[item].value)
        # return dictionary
        
    def update_params(self, **kwargs):
        self.parvals = self.params.valuesdict()
        if 'rho_w' in self.parvals:
            rho_w_value = self.parvals['rho_w']*self.e_radius
            self.parvals.update({'rho_w': rho_w_value})
        else: pass
        if 'rho_hg' in self.parvals:
            rho_hg_value = self.parvals['Z_HG']/self.parvals['v_HG']*self.e_radius
            self.parvals.update({'rho_hg': rho_hg_value})
        else: pass
        if 'rho_cg' in self.parvals:
            rho_cg_value = self.parvals['Z_CG']/self.parvals['v_CG']*self.e_radius
            self.parvals.update({'rho_cg': rho_cg_value})
        else: pass
        if 'rho_ch2' in self.parvals:
            rho_ch2_value = 8/self.parvals['v_CH2']*self.e_radius
            self.parvals.update({'rho_ch2': rho_ch2_value})
        else: pass
        if 'rho_ch3' in self.parvals:
            rho_ch3_value = 9/self.parvals['v_CH3']*self.e_radius
            self.parvals.update({'rho_ch3': rho_ch3_value})
        else: pass
        if 'scale' in self.parvals:
            scale_value = self.parvals['scale']*1e8
            self.parvals.update({'scale': scale_value})
        else: pass
        if 'background' in self.parvals:
            background_value = self.parvals['background']*1e-5
            self.parvals.update({'background': background_value})
        else: pass
    
    def load_data(self, q = [], i_q = [], err_q = [], **kwargs):
        self.q = q
        self.i_q = i_q
        if len(err_q) == 0:
            self.err_q = 0.1*i_q
        else:
            self.err_q = err_q
        self.log_q = np.log10(q)
        self.log_i_q = np.log10(i_q)
        self.log_err_q = 1/np.log10(err_q) * 1/np.log10(np.exp(1))
    
    def load_volume_distributions(self, **kwargs):
        self.p_hg_i = self.gera_p_n(self.parvals['c_hg'], -self.parvals['z_hg'], self.parvals['sigma_hg'])
        self.p_hg_o = self.gera_p_n(self.parvals['c_hg'], self.parvals['z_hg'], self.parvals['sigma_hg'])
        self.p_hg = self.p_hg_i + self.p_hg_o
        self.p_cg_i = self.gera_p_n(self.parvals['c_cg'], -self.parvals['z_cg'], self.parvals['sigma_cg'])
        self.p_cg_o = self.gera_p_n(self.parvals['c_cg'], self.parvals['z_cg'], self.parvals['sigma_cg'])
        self.p_cg = self.p_cg_i + self.p_cg_o
        self.p_hc = self.gera_p_hc()
        self.p_ch2 = self.gera_p_ch2()
        self.p_ch3 = self.gera_p_n(self.parvals['c_ch3'], self.parvals['z_ch3'], self.parvals['sigma_ch3'])
        self.p_water = self.gera_p_water()
        # Array com soma das probs para garantir que tá tudo em 1
        self.p_total = self.gera_p_soma()
        self.volume_distributions.update({'p_hg_i' : self.p_hg_i,
                                     'p_hg_o' : self.p_hg_o, 
                                     'p_hg' : self.p_hg,
                                     'p_cg_i' : self.p_cg_i,
                                     'p_cg_o' : self.p_cg_o,
                                     'p_cg' : self.p_cg,
                                     'p_hc' : self.p_hc,
                                     'p_ch2' : self.p_ch2,
                                     'p_ch3' : self.p_ch3,
                                     'p_water' : self.p_water,
                                     'p_total' : self.p_total})
        self.z_inter = self.calcula_z_inter()
        self.rho_distribution = ((self.parvals['rho_hg'] - self.parvals['rho_w']) * \
                                       self.volume_distributions['p_hg'] + \
                                (self.parvals['rho_cg'] - self.parvals['rho_w']) * \
                                      self.volume_distributions['p_cg'] + \
                                (self.parvals['rho_ch2'] - self.parvals['rho_w']) * \
                                      self.volume_distributions['p_ch2'] + \
                                (self.parvals['rho_ch3'] - self.parvals['rho_w']) * \
                                      self.volume_distributions['p_ch3']) / \
                                self.e_radius
        
    def load_r_distribution(self, **kwargs):
        self.sigma_r = self.parvals['polydispersity'] * self.parvals['r']
        self.r_array = np.linspace(self.parvals['r']-2*self.sigma_r, self.parvals['r']+2*self.sigma_r, self.sample_size)
        # Calcula distribuição
        if self.features['distribution'] == 'gaussian':
            self.distribution_array = self.calcula_gaussiana(self.parvals['r'], self.sigma_r, self.r_array)
            self.norm = np.sum(self.distribution_array)
        elif self.features['distribution'] == 'lognormal':
            self.distribution_array = self.calcula_lognormal(self.parvals['r'], self.sigma_r, self.r_array)
            self.norm = np.sum(self.distribution_array)
        elif self.features['distribution'] == 'schulz':
            self.distribution_array = self.calcula_schulz(self.parvals['r'], self.sigma_r, self.r_array)
            self.norm = np.sum(self.distribution_array)
        elif self.features['distribution'] == 'delta':
            self.sample_size = 1
            self.r_array = np.array([self.parvals['r']])
            self.distribution_array = 1
            self.norm = 1
        else:
            print('Non-existant distribution...')
        
    def load_intensities(self, **kwargs):
        self.intensities = self.calcula_intensidades(**kwargs)
        self.log_intensities = np.log10(self.intensities)
        
    # Atualiza distribuições e intensidades a partir dos parâmetros
    def load_all(self):
        self.update_params()
        self.load_r_distribution()
        self.load_volume_distributions()
        self.load_intensities()
    
    def calcula_gaussiana(self, mu, sigma, array, **kwargs):
        return 1/(np.sqrt(2*np.pi) * sigma) * np.exp(-(array - mu)**2/(2*sigma**2))

    def calcula_lognormal(self, mu, sigma, array, **kwargs):
        return 1/(np.sqrt(2*np.pi) * sigma * array) * np.exp(-(np.log(array) - mu)**2/(2*sigma**2))
    
    def calcula_schulz(self, mu, sigma, array, **kwargs):
        z = 1/sigma - 1
        return ((z+1)/mu)**(z+1)*array**z/special.gamma(z+1)*np.exp(-array*(z+1)/mu)
    
    # Recebe array de x e retorna f(x) gaussiano
    ## Aplicação aqui: calcular P_CH3 e P_p. Note que no paper não temos o fator 1/srqt(sigma**2)
    def calcula_array_gaussiano(self, mu, sigma, **kwargs):
        return (1 / np.sqrt(2*np.pi) * np.exp(-(self.z_array - mu)**2 / (2*sigma**2)))
    
    def calcula_integral_definida_gaussiana(lim_inf, lim_sup , z, sigma, **kwargs):
        return (norm.cdf(lim_inf, loc = z, scale = sigma) - norm.cdf(lim_sup , loc = z, scale = sigma))
        
    # Monta o P_n gaussiano, que é de apenas uma gaussiana
    def gera_p_n(self, c_n, mu_n, sigma_n, **kwargs):
        return np.array(c_n * self.calcula_array_gaussiano(mu_n, sigma_n))
    
    # Constrói um ponto da função da cauda, dada por condicionais que levam a sen^2 ou cos^2
    ## z_ch2 é a posição de probabilidade 0.5 e 2sigma_ch2 é a largura da oscilação
    def gera_p_hc(self, **kwargs):
        p_hc = np.zeros(len(self.z_array))
        for i in range(len(self.z_array)):
            if -self.parvals['z_ch2'] - self.parvals['sigma_ch2'] < self.z_array[i] < -self.parvals['z_ch2'] + self.parvals['sigma_ch2']:
                p_hc[i] = np.sin((self.z_array[i] - (-self.parvals['z_ch2']) + self.parvals['sigma_ch2']) / (2 * self.parvals['sigma_ch2']) * np.pi/2)**2
            elif -self.parvals['z_ch2'] + self.parvals['sigma_ch2'] < self.z_array[i] < self.parvals['z_ch2'] - self.parvals['sigma_ch2']:
                p_hc[i] = 1
            elif self.parvals['z_ch2'] - self.parvals['sigma_ch2'] < self.z_array[i] < self.parvals['z_ch2'] + self.parvals['sigma_ch2']:
                p_hc[i] = np.cos((self.z_array[i] - self.parvals['z_ch2'] + self.parvals['sigma_ch2']) / (2 * self.parvals['sigma_ch2']) * np.pi/2)**2
            else:
                pass
        return np.array(p_hc)
        
    # Calcula P_ch2 pela diferença entre P_hc e P_ch3 (que é gaussiana)
    def gera_p_ch2(self, **kwargs):
        return self.volume_distributions['p_hc'] - self.volume_distributions['p_ch3']
    
    # Calcula a distribuição da água a partir de todas as outras de modo que P_w = 1 - soma das outras
    ## Recebe lista com valores calculados para cada distribuição. Útil pra plotar P_w.
    def gera_p_water(self, **kwargs):
        all_probs = [self.volume_distributions['p_hg'], self.volume_distributions['p_cg'], \
                     self.volume_distributions['p_ch2'], self.volume_distributions['p_ch3']]
        soma = np.zeros(len(self.z_array))
        for i in range(len(all_probs)):
            soma = np.add(soma, all_probs[i])
        return 1 - soma
    
    # Calcula a soma de todas as distribuições para garantir que a normalização está OK
    ## Recebe lista com valores calculados para cada distribuição.
    def gera_p_soma(self, **kwargs):
        all_probs = [self.volume_distributions['p_hg'], self.volume_distributions['p_cg'], \
                     self.volume_distributions['p_ch2'], self.volume_distributions['p_ch3'], \
                     self.volume_distributions['p_water']]
        soma = np.zeros(len(self.z_array))
        for i in range(len(all_probs)):
            soma = np.add(soma, all_probs[i])
        return soma
    
    # Calcula a amplitude de espalhamento do termo do lipossomo para todo o range de q
    def calcula_amplitude_casca_esferica(self, r_shell, **kwargs):
        return 4. * np.pi * r_shell**2 * np.sin(self.q*r_shell) / (self.q*r_shell)
    
    # Calcula o módulo do fator de forma de casca esférica para todo o range de q
    def calcula_fator_forma_casca_esferica(self, r_shell, **kwargs):
        return np.abs(self.calcula_amplitude_casca_esferica(r_shell))**2

    # Calcula a amplitude de espalhamento do termo do lipossomo para distribuição em todo o range de q
    def calcula_amplitude_casca_esferica_distribuicao(self, **kwargs):
        casca_esferica_distribuicao = np.zeros(len(self.q))
        for i in range(len(self.r_array)):
            casca_esferica_distribuicao = casca_esferica_distribuicao + (1/self.norm) * \
                self.calcula_amplitude_casca_esferica(self.r_array[i]) * self.distribution_array[i]
        return casca_esferica_distribuicao

    # Calcula o módulo do fator de forma de casca esférica para todo o range de q
    def calcula_fator_forma_casca_esferica_distribuicao(self, **kwargs):
        casca_esferica_distribuicao = np.zeros(len(self.q))
        for i in range(len(self.r_array)):
            casca_esferica_distribuicao = casca_esferica_distribuicao + (1/self.norm) * \
                self.calcula_fator_forma_casca_esferica(self.r_array[i]) * self.distribution_array[i]
        return casca_esferica_distribuicao
    
    # Calcula o módulo do fator de forma de casca esférica com distribuição de Schulz para todo o range de q
    ## Sigma e a polidispersão relativa, como exposta no artigo do Pencer (2006)
    ## Também fiz a implementação (até agora mais correta na escala) pelo artigo do Kucerka de 2007 (Langmuir)
    def calcula_fator_forma_casca_esferica_schulz(self, **kwargs):
        z = self.parvals['r']**2/self.sigma_r**2 - 1.
        s = self.parvals['r']/self.sigma_r**2
        if self.features['which'] == 'p(q)':
            return 8.*np.pi*(z+2.)*(z+3.)/(s**2*self.q**2) * (1. - (1. + 4*self.q**2/s**2)**(-(z+1.)/2.) * s**2 * \
                                                  np.cos((3.+z) * np.arctan(2.*self.q/s)) / (4.*self.q**2 + s**2))
        elif self.features['which'] == 'f(q)':
            return 4.*np.pi*(z+1.)*(z+2.)*(z+3.)/self.q * (1. + self.q**2/s**2)**(-z/2.) * s * \
                                                  np.sin((4.+z) * np.arctan(self.q/s)) / (self.q**2 + s**2)**2
        elif (self.features['which'] == 'Kucerka' or self.features['which'] == 'kucerka'):
            return 8.*np.pi*(z+1.)*(z+2.)/(s**2*self.q**2) * (1. - (1. + 4.*self.q**2/s**2)**(-(z+3.)/2.) * \
                                                  np.cos((z+3.) * np.arctan(2.*self.q/s)))
        else:
            print('Por favor escolher p(q), f(q) ou kucerka na função calcula_fator_forma_casca_esferica_schulz!')
            raise SystemExit
    
    # Calcula o fator de forma de cadeia gaussiana para todo o range de q
    def calcula_fator_forma_cadeia_gaussiana(self, rg_dummy, **kwargs):
        chi = (self.q* rg_dummy)**2
        return 2 * (np.exp(-chi) - 1 + chi)/(chi**2)
        
    # Calcula F_cos_lip pela expressão do SI do paper
    def calcula_f_cos_lip(self, **kwargs):
        delta_rho_hg = self.parvals['rho_hg'] - self.parvals['rho_w']
        delta_rho_cg = self.parvals['rho_cg'] - self.parvals['rho_w']
        delta_rho_ch2 = self.parvals['rho_ch2'] - self.parvals['rho_w']
        delta_rho_ch3 = self.parvals['rho_ch3'] - self.parvals['rho_w']
        fator_hg = delta_rho_hg * (self.parvals['c_hg'] * self.parvals['sigma_hg'] * np.cos(self.q * (-self.parvals['z_hg'])) * \
                   np.exp(-((self.q * self.parvals['sigma_hg'])**2) / 2.)) + delta_rho_hg * (self.parvals['c_hg'] * self.parvals['sigma_hg'] * \
                   np.cos(self.q * self.parvals['z_hg']) * np.exp(-((self.q * self.parvals['sigma_hg'])**2) / 2.))
        fator_cg = delta_rho_cg * (self.parvals['c_cg'] * self.parvals['sigma_cg'] * np.cos(self.q * (-self.parvals['z_cg'])) * \
                   np.exp(-((self.q * self.parvals['sigma_cg'])**2) / 2.) + self.parvals['c_cg'] * self.parvals['sigma_cg'] * \
                   np.cos(self.q * self.parvals['z_cg']) * np.exp(- ((self.q * self.parvals['sigma_cg'])**2) / 2.))   
        fator_ch3 = 2. * delta_rho_ch3 * self.parvals['c_ch3'] * self.parvals['sigma_ch3'] * np.cos(self.q * self.parvals['z_ch3']) * \
                    np.exp(- ((self.q * self.parvals['sigma_ch3'])**2) / 2) - 2. * delta_rho_ch2 * \
                    self.parvals['c_ch3'] * self.parvals['sigma_ch3'] * np.cos(self.q * self.parvals['z_ch3']) * np.exp(- ((self.q * self.parvals['sigma_ch3'])**2) / 2)
        if self.features['ch2'] == 'nielsen':
            fator_ch2 = (np.pi**2 * delta_rho_ch2 * np.cos(self.q * self.parvals['sigma_ch2']) * np.sin(self.q * (-self.parvals['z_ch2']))) / \
                   (-np.pi**2*self.q + 4. * self.q**3 * self.parvals['sigma_ch2']**2) + \
                   (-1.)*(np.pi**2 * delta_rho_ch2 * np.cos(self.q * self.parvals['sigma_ch2']) * np.sin(self.q * self.parvals['z_ch2'])) / \
                   (-np.pi**2*self.q + 4. * self.q**3 * self.parvals['sigma_ch2']**2)
        elif self.features['ch2'] == 'katsaras':
            c_ch2 = self.parvals['v_ch2']/(self.parvals['area_lipid'] * self.parvals['sigma_ch2'])
            fator_ch2 = 2. * self.delta_rho_ch2 * c_ch2 * np.sin(self.q * self.parvals['z_ch2'])/self.q * \
                        np.exp(- ((self.q * self.parvals['sigma_ch2'])**2) / 2)
        else:
            print('Error in CH2 mathematical definition. Should be nielsen or katsaras')
            SystemExit()
        #print('Calculou f_cos_lip')
        return fator_hg + fator_cg + fator_ch2 + fator_ch3
    
    # Calcula F_sin_lip pela expressão infernal do SI do paper
    def calcula_f_sin_lip(self, **kwargs):
        delta_rho_hg = self.parvals['rho_hg'] - self.parvals['rho_w']
        delta_rho_cg = self.parvals['rho_cg'] - self.parvals['rho_w']
        delta_rho_ch2 = self.parvals['rho_ch2'] - self.parvals['rho_w']
        fator_hg = delta_rho_hg * (self.parvals['c_hg'] * self.parvals['sigma_hg'] * np.sin(self.q * (-self.parvals['z_hg'])) * \
                   np.exp(-((self.q * self.parvals['sigma_hg'])**2) / 2.)) + delta_rho_hg * (self.parvals['c_hg'] * self.parvals['sigma_hg'] * \
                   np.sin(self.q * self.parvals['z_hg']) * np.exp(-((self.q * self.parvals['sigma_hg'])**2) / 2.))
        fator_cg = delta_rho_cg * (self.parvals['c_cg'] * self.parvals['sigma_cg'] * np.sin(self.q * (-self.parvals['z_cg'])) * \
                   np.exp(-((self.q * self.parvals['sigma_cg'])**2) / 2.) + self.parvals['c_cg'] * self.parvals['sigma_cg'] * \
                   np.sin(self.q * self.parvals['z_cg']) * np.exp(- ((self.q * self.parvals['sigma_cg'])**2) / 2.))
        if self.features['ch2'] == 'nielsen':
            fator_ch2 = (1/self.q) * delta_rho_ch2 * \
                    (1. - (np.pi**2 * np.cos(self.q * self.parvals['sigma_ch2']) * np.cos(self.q * (-self.parvals['z_ch2']))) / \
                    (-np.pi**2 - 4. * self.q**2 * self.parvals['sigma_ch2']**2)) + \
                    (-1.) * (1/self.q) * delta_rho_ch2 * (1. - (np.pi**2 * np.cos(self.q * self.parvals['sigma_ch2']) * \
                    np.cos(self.q * self.parvals['z_ch2'])) / (-np.pi**2 - 4. * self.q**2 * self.parvals['sigma_ch2']**2))
        elif self.features['ch2'] == 'katsaras':
            fator_ch2 = 0
        else:
            print('Error in CH2 mathematical definition. Should be nielsen or katsaras')
            SystemExit()
        #print('Calculou f_sin_lip')
        return fator_hg + fator_cg + fator_ch2

    # Calcula fator de forma apenas da membrana |F_PB|, e não |F_PB|^2!
    def calcula_fator_forma_membrana(self, **kwargs):
        membrane = np.sqrt(self.calcula_f_cos_lip(**kwargs)**2 + self.calcula_f_sin_lip(**kwargs)**2)
        return np.array(membrane, dtype=float)
    
    # Encontra mínimo de uma função pelo método de Brent-Dekker
    ## Aplicação no código: achar z_inter, raiz da diferença entre P_ch2 e P_p
    #def calcula_z_inter(diferenca_p_ch2_e_p_p, p_ch2, p_p, z_ch3in, z_ch3ax):
        # Pensar bem nos z_ch3in e z_ch3ax pra não ter mais de uma raiz e para f(z_ch3in) e f(z_ch3ax) serem
        # de sinais diferentes, senão dá problema
    #    optimize.brentq(diferenca_p_ch2_e_p_p, z_ch3in, z_ch3ax, args=(), xtol=2e-12, \
    #           rtol=8.881784197001252e-16, maxiter=100, full_output=False, disp=True)
    
    # Encontra ponto de intersecção entre P_HC e P_P
    def calcula_z_inter(self, **kwargs):
        diferenca = self.volume_distributions['p_hc'] - self.volume_distributions['p_p']
        indice = np.where(diferenca == min(self.volume_distributions['p_hc'] - self.volume_distributions['p_p']))[0][0]
        return self.z_array[indice]
    
    # Calcula a integral de P_hc de acordo com a SI do paper
    def calcula_integral_definida_p_hc(self, **kwargs):
        k1 = np.pi / (4*self.parvals['sigma_ch2'])
        k2 = np.pi * (- (-self.parvals['z_ch2']) + self.parvals['sigma_ch2']) / (4*self.parvals['sigma_ch2'])
        #print('Passou pela função calcula_integral_definida_p_hc')
        return (2 * (k1 * (self.parvals['z_ch2'] + self.parvals['sigma_ch2']) + k2) + \
                np.sin(2 * (k1 * (self.parvals['z_ch2'] + self.parvals['sigma_ch2']) + k2))) / (4 * k1) - \
               (2 * (k1 * self.z_inter + k2) + np.sin(2 * (k1 * self.z_inter + k2))) / (4 * k1)

    # Calcula intensidade para lipossomos polidispersos com ou sem peptídeo inserido (ver flag peptide como 'yes' ou 'no')
    def calcula_intensidades(self, ch2 = 'nielsen', **kwargs):
        termo_lipossomo = self.calcula_fator_forma_casca_esferica_distribuicao()
        termo_membrana = self.calcula_fator_forma_membrana(**kwargs)
        res = self.parvals['n_scatterers'] * (termo_lipossomo * termo_membrana**2) * \
              self.parvals['scale'] + self.parvals['background']
        #print('Calculou intensidades')
        return np.array(res, dtype=float)

# ------------------------------------------------------------------ #

    def construct_parvals_dict(self, minimizer_result):
        parameters = minimizer_result.params
        dict_output = parameters.valuesdict()
        return dict_output

    # Acha o índice do valor mais próximo do procurado
    def find_nearest_index(self, array, value):
        array_subtraction = value * np.ones(len(array))
        array_difference = np.absolute(array - array_subtraction)
        minimum = np.amin(array_difference)
        return np.where(array_difference == minimum)[0][0]

    # Escolhe pontos dos dados para ajustar
    def escolhe_pontos(self, array, index_inicio, index_final):
        array_reduzido = np.zeros(index_final - index_inicio)
        for i in range(index_final - index_inicio):
            array_reduzido[i] = array[i+index_inicio]
        return array_reduzido

    def find_aberrant_probs(self, **kwargs):
        index_list = []
        z_list = []
        for i in range(len(self.z_array)):
            if self.volume_distributions['p_water'][i] < -0.01:
                print(self.volume_distributions['p_water'][i])
                index_list.append(i)
                z_list.append(self.z_array[i])
            else: pass
        return z_list, index_list
    
    # Sort z values and indexes by absolute value to work the membrane "from inside out"
    def coupled_sort_abs_z_values(self, z, index):
        z_list_abs = []
        # Sort z values and indexes by absolute value to work the membrane "from inside out"
        for j in range(len(z)):
            z_list_abs.append(abs(z[j]))
        sorting_list = np.array([z_list_abs, index, z])
        sorting_list = sorting_list[:, sorting_list[0, :].argsort()]
        z_sort_abs = sorting_list[0]
        index_sort = sorting_list[1]
        z_sort = sorting_list[2]
        return z_sort_abs, index_sort, z_sort
    
    def fitting_adjustments(self, **kwargs):
        z_list, index_list = self.find_aberrant_probs()
        z_list_abs_sorted, index_list_sorted, z_list_sorted = \
                 self.coupled_sort_abs_z_values(z_list, index_list)  
        index_list_sorted = (np.rint(index_list_sorted)).astype(int)
        for i in range(len(z_list_sorted)):
            index = index_list_sorted[i]
            if 0 > z_list_sorted[i] > -self.parvals['z_ch2'] - self.parvals['sigma_ch2']:
                while self.volume_distributions['p_cg_i'][index] > 0.01:
                    while self.volume_distributions['p_hg_i'][index] > 0.01:
                        self.params['z_hg'].value = self.params['z_hg'].value - 0.05
                        self.update_params()
                        self.load_volume_distributions()
                    self.params['z_cg'].value = self.params['z_cg'].value - 0.05
                    self.update_params()
                    self.load_volume_distributions()
            elif 0 < z_list_sorted[i] < self.parvals['z_ch2'] + self.parvals['sigma_ch2']:
                while self.volume_distributions['p_cg_o'][index] > 0.01:
                    while self.volume_distributions['p_hg'][index] > 0.01:
                        self.params['z_hg'].value = self.params['z_hg'].value + 0.05
                        self.update_params()
                        self.load_volume_distributions()
                    self.params['z_cg'].value = self.params['z_cg'].value + 0.05
                    self.update_params()
                    self.load_volume_distributions()
            else:
                pass
        z_list, index_list = self.find_aberrant_probs() # Recarrega a lista pra arrumar P_HG e P_CG
        z_list_abs_sorted, index_list_sorted, z_list_sorted = \
                 self.coupled_sort_abs_z_values(z_list, index_list) 
        while len(z_list_sorted) > 0:
            for i in range(len(z_list_sorted)):
                if z_list_sorted[i] < 0:
                    self.params['sigma_hg'].value = self.params['sigma_hg'].value - 0.05
                    self.update_params()
                    self.load_volume_distributions()
                elif z_list_sorted[i] > 0:
                    self.params['sigma_hg'].value = self.params['sigma_hg'].value - 0.05
                    self.update_params()
                    self.load_volume_distributions()
                else:
                    pass

    #-------------------AJUSTE-------------------#

    # Dá pequenas variações nos parâmetros de ajuste para avaliar reprodutibilidade
    def flutua_params(self, parametros):
        for p in parametros:
            if parametros[p].vary == True:
                parametros[p].value = parametros[p].value*np.random.uniform(0.99,1.01)
            else:
                pass

    # Faz o ajuste e retorna os resultados no formato de saída do LMFIT por Levenberg-Marquardt
    ## Com incertezas
    def ajusta_levenberg_marquardt(self, x_data, y_data, yerr_data, residuo, parametros, callback = None, passo = None):
        minimizador = Minimizer(residuo, parametros, nan_policy='omit', iter_cb = callback, \
                                epsfcn = passo, fcn_args=(x_data, y_data, yerr_data))
        return minimizador.minimize()
    
    # Faz o ajuste e retorna os resultados no formato de saída do LMFIT por Levenberg-Marquardt
    ## Sem incertezas
    def ajusta_levenberg_marquardt_no_errors(self, x_data, y_data, residuo, parametros, callback = None, passo = None):
        minimizador = Minimizer(residuo, parametros, nan_policy='omit', iter_cb = callback, \
                            epsfcn = passo, fcn_args=(x_data, y_data))
        return minimizador.minimize()

    # Faz o ajuste e retorna os resultados no formato de saída do LMFIT por Evolução Diferencial
    def ajusta_evolucao_diferencial(self, x_data, y_data, residuo, parametros, callback = None):
        minimizador = Minimizer(residuo, parametros, nan_policy='omit', iter_cb = callback, \
                            fcn_args=(x_data, y_data))
        return minimizador.minimize(method = 'differential_evolution')

    # Faz o ajuste e retorna os resultados no formato de saída do LMFIT por Nelder-Mead
    def ajusta_nelder_mead(self, x_data, y_data, residuo, parametros, callback = None):
        minimizador = Minimizer(residuo, parametros, nan_policy='omit', iter_cb = callback, \
                            fcn_args=(x_data, y_data), max_nfev = 100)
        return minimizador.scalar_minimize(method = 'Nelder-Mead')
        
    def ajusta_dual_annealing(self, x_data, y_data, residuo, parametros, callback = None):
        minimizador = Minimizer(residuo, parametros, nan_policy='omit', iter_cb = callback, \
                            fcn_args=(x_data, y_data))
        return minimizador.minimize(method = 'dual_annealing')
        
    def ajusta_basin_hopping(self, x_data, y_data, residuo, parametros, callback = None):
        minimizador = Minimizer(residuo, parametros, nan_policy='omit', iter_cb = callback, \
                            fcn_args=(x_data, y_data))
        return minimizador.minimize(method = 'basinhopping')

    # Faz o ajuste e retorna os resultados no formato de saída do LMFIT por Nelder-Mead
    def ajusta_geral(self, x_data, y_data, residuo, parametros, callback = None, \
                     passo = None, method = 'nm-lm', perturbations = False):
        if perturbations == True:
            self.flutua_params(parametros)
        else:
            pass
        if method == 'nm-lm':
            result_nm = self.ajusta_nelder_mead(x_data, y_data, residuo, parametros, callback)
            result_lm = self.ajusta_levenberg_marquardt_no_errors(x_data, y_data, residuo, \
                             result_nm.params, callback, passo)
            return result_lm
        elif method == 'de':
            result_de = self.ajusta_evolucao_diferencial(x_data, y_data, residuo, \
                                              parametros, callback)
            return result_de
        elif method == 'da':
            result_da = self.ajusta_dual_annealing(x_data, y_data, residuo, \
                                              parametros, callback)
            return result_da
        elif method == 'bh':
            result_bh = self.ajusta_basin_hopping(x_data, y_data, residuo, \
                                              parametros, callback)
            return result_bh
        elif method == 'de-lm':
            result_de = self.ajusta_evolucao_diferencial(x_data, y_data, residuo, \
                                              parametros, callback)
            result_lm = self.ajusta_levenberg_marquardt_no_errors(x_data, y_data, residuo, \
                             result_de.params, callback, passo)
            return result_lm
        else:
            SystemExit()

    # Faz o ajuste e retorna os resultados no formato de saída do LMFIT por força bruta
    ## Candidatos é o número de n tentativas salvas
    def ajusta_forca_bruta(self, x_data, y_data, yerr_data, residuo, parametros, candidatos):
        minimizador = Minimizer(residuo, parametros, fcn_args=(x_data, y_data, yerr_data))
        # Ns é usado quando brute_step não é especificado
        resultado = minimizador.minimize(method='brute', Ns=5, keep=candidatos)
        # Mostra o primeiro candidato em forma de lista
        print(resultado.brute_x0)
        return resultado        

