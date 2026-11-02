# Tech Challenge — Fases 1 e 2

Projeto acadêmico da **Pós-graduação IA para Devs (FIAP)**:

- **Fase 1:** classificação de câncer de mama e SOP (PCOS), avaliação e explicabilidade.
- **Fase 2 — Projeto 1:** otimização genética dos modelos e interpretação agregada por LLM.
- **Fase 2 — Projeto 2:** otimização de rotas para distribuição simulada de medicamentos e insumos.

## Manual e início rápido

Consulte o [MANUAL_TECNICO.md](./MANUAL_TECNICO.md) para preparar o ambiente,
gerar os resultados e iniciar a aplicação.

Para iniciar:

```powershell
git clone https://github.com/rsjtecnologia/tech-challenge-fase1-1.git
Set-Location .\tech-challenge-fase1-1
. .\pre_execucao.ps1
.\run_notebooks.ps1
```

Veja no manual como gerar os resultados, configurar o Gemini e iniciar web/API.
O treino e os algoritmos genéticos rodam localmente; Gemini é usado para gerar
interpretações e instruções em linguagem natural.

> Uso acadêmico: não substitui diagnóstico médico nem valida uma operação real
> de distribuição hospitalar. Não envie dados identificáveis de pacientes à LLM.
