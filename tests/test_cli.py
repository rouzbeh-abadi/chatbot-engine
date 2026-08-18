from rabadi_ae_afa_3_5.cli import main


def test_main_greets(capsys):
    main()
    assert "rabadi-ae-afa-3-5" in capsys.readouterr().out
