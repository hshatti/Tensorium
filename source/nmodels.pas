unit nModels;
{$ifdef fpc}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils
  , ntensors
  , ntypes
  , nBaseLayer
  , nnet
  , nConnectedlayer
  , nLogisticLayer
  , nSoftmaxLayer
  , nCostLayer
  , nConvolutionLayer
  , nMaxPoolLayer
  , nDropOutLayer
  ;

type
  TNetCfg =record
    a: integer;
    b:shortstring
  end;

function simpleDenseMNIST:TArray<TBaseLayer>;
function leNetMNIST:TArray<TBaseLayer>;
function leNetCIFAR10:TArray<TBaseLayer>;
function deepCIFAR10:TArray<TBaseLayer>;

implementation

function simpleDenseMNIST:TArray<TBaseLayer>;
begin
  result := [
        TConnectedLayer.Create(1, 1, 28*28, 64, acRELU{, true})
      , TConnectedLayer.Create(1, 1, 64   , 64, acRELU{, true})
      , TConnectedLayer.Create(1, 1, 64   , 32, acRELU{, true})
      , TConnectedLayer.Create(1, 1, 32   , 32, acRELU{, true})
      , TConnectedLayer.Create(1, 1, 32   , 10, acLINEAR{, true})
      , TSoftmaxLayer.Create(1,10)
      //, TLogisticLayer.Create(1,10)
      //, TCostLayer.Create(1,10,ctSSE,1)
    ]
end;

function leNetMNIST:TArray<TBaseLayer>;
begin
  result := [
        TConvolutionalLayer.Create(1, 28, 28, 1, 6, 1, 5, 1, 1, 1, 2, acReLU)
      , TMaxPoolLayer.Create(1, 28, 28,  6, 2)
      , TConvolutionalLayer.Create(1, 14, 14, 6, 16, 1, 5, 1, 1, 1, 0, acReLU)
      , TMaxPoolLayer.Create(1, 10, 10, 16, 2)
      , TConvolutionalLayer.Create(1, 5, 5, 16, 120, 1, 5, 1, 1, 1, 0, acReLU)
      , TConnectedLayer.Create(1, 1, 120   , 84, acReLU)
      , TConnectedLayer.Create(1, 1, 84   , 10, acLINEAR)
      , TSoftmaxLayer.Create(1,10)
      //, TLogisticLayer.Create(1,10)
      //, TCostLayer.Create(1,10,ctSSE,1)
    ]
end;

function leNetCIFAR10:TArray<TBaseLayer>;
begin
  result := [
        TConvolutionalLayer.Create(1, 32, 32, 3, 6, 1, 5, 1, 1, 1, 0, acReLU, true)
      , TMaxPoolLayer.Create(1, 28, 28,  6, 2)
      , TConvolutionalLayer.Create(1, 14, 14, 6, 12, 1, 5, 1, 1, 1, 0, acReLU, true)
      , TMaxPoolLayer.Create(1, 10, 10, 12, 2)
      , TConvolutionalLayer.Create(1, 5, 5, 12, 120, 1, 5, 1, 1, 1, 0, acReLU, true)
      , TConnectedLayer.Create(1, 1, 120 , 84, acReLU)
      , TConnectedLayer.Create(1, 1, 84 , 10, acLINEAR)
      , TSoftmaxLayer.Create(1,10)
      //, TLogisticLayer.Create(1,10)
      //, TCostLayer.Create(1,10,ctSSE,1)
    ]
end;

function deepCIFAR10:TArray<TBaseLayer>;
begin
  result := [
      TConvolutionalLayer.Create(1, 32, 32, 3, 32, 1, 3, 1, 1, 1, 1, acRELU, true),
      TConvolutionalLayer.Create(1, 32, 32, 32, 32, 1, 3, 1, 1, 1, 1, acRELU, true),
      TMaxPoolLayer.Create(1, 32, 32, 32, 2),

      TConvolutionalLayer.Create(1, 16, 16, 32, 64, 1, 3, 1, 1, 1, 1, acRELU, true),
      TConvolutionalLayer.Create(1, 16, 16, 64, 64, 1, 3, 1, 1, 1, 1, acRELU, true),
      TMaxPoolLayer.Create(1, 16, 16, 64, 2),

      TConvolutionalLayer.Create(1, 8, 8, 64, 128, 1, 3, 1, 1, 1, 1, acRELU, true),
      TConvolutionalLayer.Create(1, 8, 8, 128, 128, 1, 3, 1, 1, 1, 1, acRELU, true),
      TMaxPoolLayer.Create(1, 8, 8, 128, 2),

      //TDropoutLayer.create(1, 0.2, 4* 4*128),

      //hidden layer
      TConnectedLayer.create(1, 1, 4*4*128, 1024, acRELU),
      //TDropoutLayer.create(1, 0.2, 1024),

      // last hidden layer i.e.. output layer
      TConnectedLayer.create(1, 1, 1024, 10, acLINEAR),
      TSoftmaxLayer.Create(1, 10)

    ]
end;

initialization

end.

